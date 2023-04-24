from __future__ import annotations

import re

from pathlib import Path
from typing import Tuple, Union, Dict, Any, BinaryIO

import h5py
import hdf5plugin
import numpy
from numpy.typing import NDArray, ArrayLike

from mxio import DataSet, XYPair, Geometry

__all__ = [
    "HDF5DataSet",
    "hdf5_file_parts",
    "CONVERTERS",
    "NUMBER_FORMATS"
]

NUMBER_FORMATS = {
    'uint16': numpy.dtype(numpy.int16),
    'uint32': numpy.dtype(numpy.int32),
    'uint64': numpy.dtype(numpy.int64),
    'int32': numpy.dtype(numpy.int32),
    'int64': numpy.dtype(numpy.int32),
}

def save_array(name, data):
    """
    Save an array to hdf5 using BitShuffle
    :param name: Name of array. Filename will be suffixed with ".h5"
    :param data: Numpy array to save
    """
    with h5py.File(f'{name}.h5', 'w') as fobj:
        fobj.create_dataset(name, data=data, **hdf5plugin.Bitshuffle(nelems=0, lz4=True))


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

    file: h5py.File
    data_sections: Dict[str, range]
    format: str
    omega_field: str = '/entry/sample/goniometer/omega'
    array_fields: tuple = ('two_theta', 'omega', 'chi', 'phi', 'kappa', 'geometry')
    header_fields: dict = {
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
        # 'geometry': (
        #     '/entry/instrument/detector/geometry/orientation/value',
        # )
    }
    oscillation_fields: dict = {
        'start': '/entry/sample/goniometer/{}',
        'delta': '/entry/sample/goniometer/{}_range_average'
    }
    format: str = 'HDF5'
    start_angles: ArrayLike

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        magic = b'\211HDF\r\n\032\n'
        if file.read(len(magic)) == magic:
            return "HDF5 Area Detector Data",

    def setup(self):
        image_path = self.directory.joinpath(self.reference)
        self.index, self.reference, self.directory = hdf5_file_parts(image_path)
        pattern = re.compile(
            r'^(?P<name>.+?)(?P<separator>[._-]?)(?P<field>((data_)?\d{4,})|(master))\.(?P<extension>(\w+))'
        )
        matched = pattern.match(self.reference)
        if matched:
            params = matched.groupdict()
            self.name = params['name']
            self.reference = f'{self.name}{params["separator"]}master.{params["extension"]}'
            self.template = f'{self.reference}/{{field:>06}}'
            self.glob = self.reference.replace('master', '??????')

        self.file = h5py.File(self.directory.joinpath(self.reference), 'r')
        self.format = 'HDF5'

        # count frames from actual data arrays
        frame_count = 0
        section_keys = list(self.file['/entry/data'].keys())
        self.data_sections = {}
        for section in section_keys:
            attrs = self.file[f'/entry/data/{section}'].attrs
            self.data_sections[section] = range(attrs['image_nr_low'], attrs['image_nr_high']+1)
            frame_count += len(self.data_sections[section])

        self.series = numpy.arange(frame_count) + 1
        self.size = frame_count
        self.get_frame(self.index)

    def extract_field(self, field: str, array: bool = False, index: slice | int = ()) -> Any:
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

    def parse_header(self, header_fields: dict, oscillation_fields: dict):
        header = {'filename': self.reference}

        for key, field in header_fields.items():
            converter = CONVERTERS.get(key, lambda v: v)
            try:
                if not isinstance(field, (tuple, list)):
                    header[key] = converter(self.extract_field(field, array=(key in self.array_fields)))
                else:
                    header[key] = tuple(
                        converter(self.extract_field(sub_field, array=(key in self.array_fields)))
                        for sub_field in field
                    )
            except (ValueError, KeyError) as err:
                pass

        for axis in ['omega', 'phi', 'chi', 'kappa']:
            try:
                start_angles = self.extract_field(oscillation_fields['start'].format(axis), array=True)
                value_varies = (start_angles.mean() != 0.0 and numpy.diff(start_angles).sum() != 0)
                if len(start_angles) == 1 or value_varies:
                    self.start_angles = start_angles

                    # found the right axis
                    for field, path in oscillation_fields.items():
                        header[f'{field}_angle'] = self.extract_field(path.format(axis), array=True)

                    header['start_angle'] = float(start_angles[0])
                    header['delta_angle'] = float(header['delta_angle'][0])
                    header['two_theta'] = 0.0
                    break
            except KeyError:
                pass

        header['size'] = XYPair(*header["size"])
        header['pixel_size'] = XYPair(*header['pixel_size'])
        header['center'] = XYPair(*header["center"])

        # detector_orientation = header['geometry'][0]
        # header['geometry'] = Geometry(
        #     detector=(tuple(detector_orientation[:3]), tuple(detector_orientation[3:])),
        #     goniometer=(-1, 0, 0), beam=(0, 0, 1)
        # )

        return header

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        index, reference, directory = hdf5_file_parts(filename)
        assert (reference, directory) == (self.reference, self.directory), "Invalid data archive"
        assert index in self.series, f"Frame {index} does not exist in this archive"

        section_name, frame_index = "", -1
        for name, frame_range in self.data_sections.items():
            if index in frame_range:
                section_name = name
                frame_index = frame_range.index(index)
                break

        header = self.parse_header(self.header_fields, self.oscillation_fields)
        header['format'] = self.format
        header['start_angle'] = self.start_angles[index-1]

        assert section_name in self.data_sections, f"Section not found"

        key = f'/entry/data/{section_name}'
        link = self.file.get(key, getlink=True)
        section_file = link.filename
        path = self.directory.joinpath(section_file)

        assert path.exists(), f"External data link file {section_file} could not be found"
        raw_data = self.extract_field(key, array=True, index=frame_index)
        data = raw_data.view(NUMBER_FORMATS[str(raw_data.dtype)])

        return header, data

