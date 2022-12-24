import re
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import h5py
import numpy
from numpy.typing import NDArray

from .hdf5 import HDF5DataSet, hdf5_file_parts

HEADERS_FIELDS = {
    'detector': '/entry/instrument/detector/description',
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
    'cutoff_value': '/entry/instrument/detector/saturation_value',
    'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
    'serial_number': '/entry/instrument/detector/serial_number',
    'size': (
        '/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
        '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'
    ),
}
OMEGA_FIELD = '/entry/data/omega'
ARRAY_FIELDS = [
    'two_theta', 'omega', 'chi', 'phi', 'kappa'
]
DATA_FIELD = '/entry/data/data'
OSCILLATION_FIELDS = {
    'start': '/entry/sample/transformations/{}',
    'delta': '/entry/sample/transformations/{}_increment_set'
}


class NXSDataSet(HDF5DataSet):

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        if extension.lower() == '.nxs':
            return "NEXUS",

    def setup(self):
        image_path = self.directory.joinpath(self.reference)
        self.index, self.reference, self.directory = hdf5_file_parts(image_path)
        pattern = re.compile(
            r'^(?P<name>.+?).(?P<extension>(\w+))'
        )
        matched = pattern.match(self.reference)
        if matched:
            params = matched.groupdict()
            self.name = params['name']
            self.template = f'{self.reference}/{{field:>06}}'

        self.file = h5py.File(self.directory.joinpath(self.reference), 'r')

        frame_count = self.extract_field(OMEGA_FIELD, array=True).size
        self.series = numpy.arange(frame_count) + 1
        self.get_frame(self.index)

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        index, reference, directory = hdf5_file_parts(filename)
        assert (reference, directory) == (self.reference, self.directory), "Invalid data archive"
        assert index in self.series, f"Frame {index} does not exist in this archive"

        header = self.parse_header(HEADERS_FIELDS, OSCILLATION_FIELDS)
        header['format'] = 'NXmx'

        data = self.extract_field(DATA_FIELD, array=True, index=index)
        return header, data
