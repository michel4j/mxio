import re

from typing import BinaryIO, Tuple

import h5py
import numpy

from .hdf5 import HDF5DataSet, hdf5_file_parts, NUMBER_FORMATS


class NXSDataSet(HDF5DataSet):
    oscillation_fields = {
        'start': '/entry/sample/transformations/{}',
        'delta': '/entry/sample/transformations/{}_increment_set'
    }
    header_fields = {
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
        # 'geometry': (
        #     '/entry/instrument/detector/geometry/orientation/value',
        # )
    }
    omega_field = '/entry/sample/transformations/omega'

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        if extension.lower() == '.nxs':
            return "NEXUS",
        else:
            try:
                f = h5py.File(file, 'r')
                value = f['/entry/definition'][()]
            except Exception:
                # deliberately ignoring any errors during file type identification.
                return ()
            else:
                return "NEXUS",

    def setup(self):
        image_path = self.directory.joinpath(self.reference)
        self.index, self.reference, self.directory = hdf5_file_parts(image_path)
        pattern = re.compile(
            r'^(?P<name>.+?)(?P<separator>[._-])?(?P<field>((data_)?\d{4,})|(master))?\.(?P<extension>(\w+))'
        )
        matched = pattern.match(self.reference)
        if matched:
            params = matched.groupdict()
            if params['extension'] == 'nxs':
                params['separator'] = '_'
            self.name = params['name']
            self.reference = f'{self.name}{params["separator"]}master.h5'
            self.template = f'{self.reference}/{{field:>06}}'
            self.glob = self.reference.replace('master', '??????')

        self.file = h5py.File(self.directory.joinpath(self.reference), 'r')

        # count frames from actual data arrays
        frame_count = 0
        section_keys = [
            section for section in self.file['/entry/data'].keys() if re.match(r'data_\d{4,}', section)
        ]
        self.data_sections = {}
        for section in section_keys:
            attrs = self.file[f'/entry/data/{section}'].attrs
            self.data_sections[section] = range(attrs['image_nr_low'], attrs['image_nr_high']+1)
            frame_count += len(self.data_sections[section])

        self.format = 'NXmx'
        self.series = numpy.arange(frame_count) + 1
        self.size = frame_count
        self.series = numpy.arange(frame_count) + 1

        self.get_frame(self.index)
