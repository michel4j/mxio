import os
import cv2
import h5py
import hdf5plugin
import numpy
import iso8601

import re
import struct

from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Sequence, Any
from numpy.typing import NDArray

from mxio.dataset import DataSet, XYPair

__all__ = [
    "HDF5DataSet"
]

HEADERS = {
    'detector': '/entry/instrument/detector/description',
    'serial_number': '/entry/instrument/detector/detector_number',
    'two_theta': '/entry/instrument/detector/goniometer/two_theta',
    'pixel_size': (
        '/entry/instrument/detector/x_pixel_size',
        '/entry/instrument/detector/y_pixel_size',
    ),
    'exposure': '/entry/instrument/detector/count_time',
    'wavelength': '/entry/instrument/beam/incident_wavelength',
    'distance': '/entry/instrument/detector/detector_distance',
    'center': (
        '/entry/instrument/detector/beam_center_x',
        '/entry/instrument/detector/beam_center_y'
    ),
    'cutoff_value': '/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff',
    'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
    'size': (
        '/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
        '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'
    ),
}
OMEGA_FIELD = '/entry/sample/goniometer/omega'
MASK_FIELD = '/entry/instrument/detector/detectorSpecific/pixel_mask'
ARRAY_FIELDS = [
    'two_theta', 'omega', 'chi', 'phi', 'kappa'
]
OSCILLATION_FIELDS = {
    'start': '/entry/sample/goniometer/{}',
    'delta': '/entry/sample/goniometer/{}_range_average'
}


def save_array(name, data):
    """
    Save an array to hdf5 using BitShuffle
    :param name: Name of array. Filename will be suffixed with ".h5"
    :param data: Numpy array to save
    """
    with h5py.File(f'{name}.h5', 'w') as fobj:
        fobj.create_dataset(name, data=data, **hdf5plugin.Bitshuffle(nelems=0, lz4=True))


def convert_date(text):
    """
    Convert ISO formatted date time into datetime object
    """
    try:
        return datetime.fromisoformat(text.decode('utf-8'))
    except (AttributeError, TypeError, ValueError):
        return iso8601.parse_date(text.decode('utf-8'))


CONVERTERS = {
    'detector': lambda v: v.decode('utf-8'),
    'serial_number': lambda v: v.decode('utf-8'),
    'two_theta': lambda v: float(v[0]),
    'pixel_size': lambda v: float(v) * 1000,
    'exposure': float,
    'wavelength': float,
    'distance': lambda v: float(v) * 1000,
    'center': float,
    'cutoff_value': int,
    'sensor_thickness': lambda v: float(v) * 1000,
    'size': int,
}


def hdf5_file_parts(image_path: Union[Path, str]) -> Tuple[int, str, Path]:
    """
    Split and HDF5 path into index, file_name and directory
    :param image_path: Path
    :return: Tuple[index, file_name, directory]
    """

    image_path = Path(image_path)
    if re.match(r'^\d+$', image_path.name):
        return int(image_path.name), image_path.parent.name, image_path.parent.parent
    else:
        return 1, image_path.name, image_path.parent


class HDF5DataSet(DataSet):
    magic = (
        {'offset': 0, "magic": b'\211HDF\r\n\032\n', "name": "HDF5 Area Detector Data"},
    )

    file: h5py.File
    data_sections: Sequence[str]
    max_section_size: int

    def setup(self):
        image_path = self.directory.joinpath(self.reference)
        self.index, self.reference, self.directory = hdf5_file_parts(image_path)
        pattern = re.compile(
            r'^(?P<name>.+?)(?P<separator>[._-]?)(?P<field>((data_)?\d+)|(master))\.(?P<extension>(\w+))'
        )
        matched = pattern.match(self.reference)
        if matched:
            params = matched.groupdict()
            self.name = params['name']
            self.reference = f'{self.name}{params["separator"]}master.{params["extension"]}'
            self.template = f'{self.reference}/{{field:>06}}'

        self.file = h5py.File(self.directory.joinpath(self.reference), 'r')

        frame_count = self.extract_field(OMEGA_FIELD, array=True).size
        self.series = numpy.arange(frame_count) + 1
        self.data_sections = list(self.file['/entry/data'].keys())
        self.max_section_size = int(numpy.ceil(frame_count / len(self.data_sections)))
        self.get_frame(self.index)

    def extract_field(self, field: str, array: bool = False, index: slice = ()) -> Any:
        """
        Extract an HDF5 field from the archive
        :param field: field path
        :param array: bool
        :param index: slice to extract a subset of a larger array
        """
        field_value = self.file[field]
        raw_value = field_value[index]
        if field_value.shape == () and array:
            raw_value = numpy.array([raw_value])
        return raw_value

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        index, reference, directory = hdf5_file_parts(filename)
        assert (reference, directory) == (self.reference, self.directory), "Invalid data archive"
        assert index in self.series, f"Frame {index} does not exist in this archive"

        header = {
            'format': 'HDF5',
            'filename': self.reference,
        }
        for key, field in HEADERS.items():
            converter = CONVERTERS.get(key, lambda v: v)
            try:
                if not isinstance(field, (tuple, list)):
                    header[key] = converter(self.extract_field(field, array=(key in ARRAY_FIELDS)))
                else:
                    header[key] = tuple(
                        converter(self.extract_field(sub_field, array=(key in ARRAY_FIELDS)))
                        for sub_field in field
                    )
            except (ValueError, KeyError) as err:
                pass

        for axis in ['omega', 'phi', 'chi', 'kappa']:
            try:
                start_angles = self.extract_field(OSCILLATION_FIELDS['start'].format(axis), array=True)
                value_varies = (start_angles.mean() != 0.0 and numpy.diff(start_angles).sum() != 0)
                if len(start_angles) == 1 or value_varies:
                    # found the right axis
                    for field, path in OSCILLATION_FIELDS.items():
                        header[f'{field}_angle'] = self.extract_field(path.format(axis), array=True)

                    header['start_angle'] = float(header['start_angle'][0])
                    header['delta_angle'] = float(header['delta_angle'][0])
                    header['two_theta'] = 0.0
                    break
            except KeyError:
                pass

        header['size'] = XYPair(*header["size"])
        header['pixel_size'] = XYPair(*header['pixel_size'])
        header['center'] = XYPair(*header["center"])

        section_index, frame_index = divmod(index, self.max_section_size)
        assert section_index < len(self.data_sections), f"Section {section_index} does not exist"
        section_name = self.data_sections[section_index]

        key = f'/entry/data/{section_name}'
        link = self.file.get(key, getlink=True)
        section_file = link.filename
        path = self.directory.joinpath(section_file)

        assert path.exists(), f"External data link file {section_file} could not be found"
        data = self.extract_field(key, array=True, index=frame_index)

        return header, data

