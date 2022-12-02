import os
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import iso8601
import numpy
import pytz

from . import DataSet
from ..log import get_module_logger

# Configure Logging
logger = get_module_logger('mxio')

HEADERS = {
    'HDF5': {
        'detector_type': '/entry/instrument/detector/description',
        'two_theta': '/entry/instrument/detector/goniometer/two_theta',
        'pixel_size': '/entry/instrument/detector/x_pixel_size',
        'exposure_time': '/entry/instrument/detector/count_time',
        'exposure_period': '/entry/instrument/detector/frame_time',
        'wavelength': '/entry/instrument/beam/incident_wavelength',
        'date': '/entry/instrument/detector/detectorSpecific/data_collection_date',
        'distance': '/entry/instrument/detector/detector_distance',
        'beam_center': ('/entry/instrument/detector/beam_center_x', '/entry/instrument/detector/beam_center_y'),
        'saturated_value': '/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff',
        'energy': '/entry/instrument/beam/incident_wavelength',
        'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
        'serial_number': '/entry/instrument/detector/detector_number',
        'detector_size': ('/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                          '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'),
    },
    'NXmx': {
        'detector_type': '/entry/instrument/detector/description',
        'two_theta': '/entry/instrument/detector/goniometer/two_theta',
        'pixel_size': '/entry/instrument/detector/x_pixel_size',
        'exposure_time': '/entry/instrument/detector/count_time',
        'wavelength': '/entry/instrument/beam/incident_wavelength',
        'date': '/entry/start_time',
        'distance': '/entry/instrument/detector/detector_distance',
        'beam_center': ('/entry/instrument/detector/beam_center_x', '/entry/instrument/detector/beam_center_y'),
        'energy': '/entry/instrument/beam/incident_wavelength',
        'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
        'detector_size': ('/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                          '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'),
    },
}


def save_array(name, data):
    """
    Save an array to hdf5 using BitShuffle
    :param name: Name of array. Filename will be suffixed with ".h5"
    :param data: Numpy array to save
    """
    with h5py.File(f'{name}.h5', 'w') as fobj:
        fobj.create_dataset(name, data=data, **hdf5plugin.Bitshuffle(nelems=0, lz4=True))


def wavelength_to_energy(wavelength):
    """
    Convert wavelength in angstroms to energy in keV.
    """
    if wavelength == 0.0:
        return 0.0
    return 12398.419300923944 / (wavelength * 1000.0)


def convert_date(text):
    """
    Convert ISO formatted date time into datetime object
    """
    try:
        return datetime.fromisoformat(text.decode('utf-8'))
    except AttributeError:
        return iso8601.parse_date(text.decode('utf-8'))


CONVERTERS = {
    'detector_type': lambda v: v.decode('utf-8'),
    'serial_number': lambda v: v.decode('utf-8'),
    'date': convert_date,
    'two_theta': lambda v: float(v[0]),
    'pixel_size': lambda v: float(v) * 1000,
    'exposure_time': float,
    'exposure_period': float,
    'wavelength': float,
    'distance': lambda v: float(v) * 1000,
    'beam_center': float,
    'saturated_value': int,
    'num_frames': int,
    'energy': wavelength_to_energy,
    'sensor_thickness': lambda v: float(v) * 1000,
    'detector_size': int,
}

ARRAY_FIELDS = [
    'two_theta', 'omega', 'chi', 'phi', 'kappa'
]

DEFAULTS = {
    'two_theta': 0.0
}

OSCILLATION_FIELDS = {
    'HDF5': {
        'start': '/entry/sample/goniometer/{}',
        'total': '/entry/sample/goniometer/{}_range_total',
        'delta': '/entry/sample/goniometer/{}_range_average'
    },
    'NXmx': {
        'start': '/entry/sample/transformations/{}',
        'delta': '/entry/sample/transformations/{}_increment_set'
    },
}

NUMBER_FORMATS = {
    'uint16': numpy.int16,
    'uint32': numpy.int32,
}


def get_section_name(kind, key):
    if kind == 'NXmx':
        return key.split('_')[-1]
    return key


class HDF5DataSet(DataSet):
    name = 'Hierarchical Data Format (version 5) data'

    def __init__(self, path, header_only=False):
        super(HDF5DataSet, self).__init__()
        self.hdf_type = 'HDF5'
        self.section_prefix = ''

        image_path = Path(path)
        if re.match(r'^\d+$', image_path.name):
            self.current_frame = int(image_path.name)
            filename = image_path.parent.name
            directory = str(image_path.parent.parent)
        else:
            self.current_frame = None
            filename = image_path.name
            directory = str(image_path.parent)

        self.directory = directory
        self.current_section = None
        self.ref_section = None
        self.section_names = []

        p0 = re.compile('^(?P<root_name>.+)_master\.h5$')
        p1 = re.compile('^(?P<root_name>.+?)_(?P<section>(?:data_)?\d+)\.h5')
        m0 = p0.match(filename)
        m1 = p1.match(filename)

        if m0:
            params = m0.groupdict()
            self.root_name = params['root_name']
            self.master_file = os.path.join(self.directory, params['root_name'] + '_master.h5')
        elif m1:
            params = m1.groupdict()
            self.master_file = os.path.join(self.directory, params['root_name'] + '_master.h5')
            self.root_name = params['root_name']
            self.ref_section = params['section']
        else:
            self.master_file = path
            self.root_name = os.path.splitext(filename)[0]

        self.name = self.root_name
        self.raw = h5py.File(self.master_file, 'r')
        self.mask = None
        self.data = None
        self.read_dataset()

    def extract_field(self, field, array=False):
        field_value = self.raw[field]
        raw_value = field_value[()]
        if field_value.shape == () and array:
            raw_value = numpy.array([raw_value])
        return raw_value

    def read_dataset(self):
        self.header = {}
        try:
            self.hdf_type = self.raw['/entry/definition'][()].decode('utf-8')
            self.section_prefix = 'data_'
        except (ValueError, KeyError):
            self.hdf_type = 'HDF5'
            self.section_prefix = ''

        header_fields = HEADERS[self.hdf_type]
        for key, field in header_fields.items():
            converter = CONVERTERS.get(key, lambda v: v)
            try:
                if not isinstance(field, (tuple, list)):
                    self.header[key] = converter(self.extract_field(field, array=(key in ARRAY_FIELDS)))
                else:
                    self.header[key] = tuple(
                        converter(self.extract_field(sub_field, array=(key in ARRAY_FIELDS)))
                        for sub_field in field
                    )
            except (ValueError, KeyError):
                logger.warning('Field corresponding to {} not found!'.format(key))
                self.header[key] = DEFAULTS.get(key)

        self.header['name'] = self.name
        self.header['format'] = self.hdf_type
        self.header['filename'] = os.path.basename(self.master_file)

        # try to find oscillation axis and parameters as first non-zero average
        oscillation_fields = OSCILLATION_FIELDS[self.hdf_type]

        for axis in ['omega', 'phi', 'chi', 'kappa']:
            try:
                self.start_angles = self.extract_field(oscillation_fields['start'].format(axis), array=True)
                value_varies = (self.start_angles.mean() != 0.0 and numpy.diff(self.start_angles).sum() != 0)
                if len(self.start_angles) == 1 or value_varies:
                    # found the right axis
                    self.header['rotation_axis'] = axis
                    for field, path in OSCILLATION_FIELDS[self.hdf_type].items():
                        print(field, path.format(axis))
                        self.header[f'{field}_angle'] = self.extract_field(path.format(axis), array=True)

                    # start angles are always sequences
                    self.header['start_angle'] = float(self.start_angles[0])

                    if self.hdf_type == 'NXmx':
                        self.header['total_angle'] = float(numpy.sum(self.header['delta_angle']))
                        self.header['delta_angle'] = float(self.header['delta_angle'][0])

                    break
            except KeyError:
                pass

        self.section_names = list(self.raw['/entry/data'].keys())

        # guess frame number ranges from number sections instead of reading external links
        num_frames = len(self.extract_field('/entry/sample/goniometer/omega', array=True))
        num_sections = len(self.section_names)
        max_frames_per_section = int(numpy.ceil(num_frames / num_sections))
        frames = numpy.arange(max_frames_per_section * num_sections) + 1
        section_frames = frames.reshape((num_sections, -1))

        self.sections = {
            self.section_names[i]: (f[0], min(f[-1], num_frames))
            for i, f in enumerate(section_frames)
        }

        # for name, d in self.raw['/entry/data'].items():
        #     if d is not None and name.startswith('data_'):
        #         self.sections[name] = (d.attrs['image_nr_low'], d.attrs['image_nr_high'])

        # NXmx data names don't match file pattern 'data_' prefix is missing
        if self.ref_section:
            self.current_section = f'{self.section_prefix}{self.ref_section}'
            self.current_frame = int(self.sections[self.current_section][0])
        else:
            if self.current_frame is None:
                self.current_section = self.section_names[0]
                self.current_frame = int(self.sections[self.current_section][0])
            else:
                for section_name, section_limits in self.sections.items():
                    if section_limits[0] <= self.current_frame <= section_limits[1]:
                        self.current_section = section_name
                        break

        width = 6
        template = '{root_name}_{{field}}.h5'.format(root_name=self.root_name)
        prefix = {'HDF5': 'data_', 'NXmx': ''}[self.hdf_type]
        regex = r'^{root_name}_{prefix}(\d{{{width}}}).h5$'.format(width=width, root_name=self.root_name, prefix=prefix)
        self.header['dataset'] = {
            'name': '{root_name}_master.h5'.format(root_name=self.root_name),
            'start_time': self.header['date'].replace(tzinfo=pytz.utc),
            'label': self.root_name,
            'directory': self.directory,
            'template': template.format(field='?' * width),
            'regex': regex,
            'start_angle': self.header['start_angle'],
            'sequence': frames,
            'current': self.current_frame
        }
        return self.read_image()

    def read_image(self):
        if self.mask is None:
            self.mask = numpy.invert(
                self.extract_field('/entry/instrument/detector/detectorSpecific/pixel_mask').view(bool)
            )

        folder = Path(self.directory)
        key = f'/entry/data/{self.current_section}'
        path = folder.joinpath(f'{self.root_name}_{self.current_section}.h5')

        if path.exists():
            # wait for file to be written, up to 10 seconds. Assume mtime > 0.1 sec means done writing
            end_time = time.time() + 10
            while time.time() - path.stat().st_mtime < 0.1 and time.time() < end_time:
                time.sleep(0.1)

            section = self.extract_field(key)
            frame_index = max(self.current_frame - self.sections[self.current_section][0], 0)
            raw_data = section[frame_index]
            data_type = NUMBER_FORMATS[str(raw_data.dtype)]
            data = raw_data.view(data_type)

            # use a  quarter of the image as a subset of data or fast analysis
            stats_subset = data[:data.shape[0] // 2, :data.shape[1] // 2]
            valid = self.mask[:data.shape[0] // 2, :data.shape[1] // 2]
            self.stats_data = stats_subset[valid]

            self.header['average_intensity'], self.header['std_dev'] = numpy.ravel(cv2.meanStdDev(self.stats_data))
            self.header['min_intensity'] = 0
            self.header['max_intensity'] = float(self.stats_data.max())  # Estimate
            self.header['overloads'] = 4 * (self.stats_data == self.header['saturated_value']).sum()  # Fast estimate
            self.header['frame_number'] = self.current_frame
            self.data = data
            return True

    def get_frame(self, index=1):
        """
        Load a specific frame
        :param index: frame index
        :return:
        """
        for section_name, section_limits in self.sections.items():
            if section_limits[0] <= index <= section_limits[1]:
                self.current_frame = index
                self.current_section = section_name
                self.header['start_angle'] = self.start_angles[self.current_frame - 1]
                return self.read_image()
        return False

    def next_frame(self):
        """Load the next frame in the dataset"""

        next_frame = self.current_frame + 1
        section_limits = self.sections[self.current_section]
        if next_frame <= section_limits[1]:
            self.current_frame = next_frame
        else:
            # skip to first image of next section
            i = self.section_names.index(self.current_section) + 1
            if i <= len(self.section_names) - 1:
                next_section = self.section_names[i]
                next_frame = self.sections[next_section][0]

                self.current_frame = next_frame
                self.current_section = next_section
            else:
                return False
        self.header['start_angle'] = float(self.start_angles[self.current_frame - 1])
        return self.read_image()

    def prev_frame(self):
        """Load the previous frame in the dataset"""

        next_frame = self.current_frame - 1
        section_limits = self.sections[self.current_section]
        if next_frame >= section_limits[0]:
            self.current_frame = next_frame
        else:
            # skip to last image of prev section
            i = self.section_names.index(self.current_section) - 1
            if i >= 0:
                next_section = self.section_names[i]
                next_frame = self.sections[next_section][1]

                self.current_frame = next_frame
                self.current_section = next_section

            else:
                return False
        self.header['start_angle'] = self.start_angles[self.current_frame - 1]
        return self.read_image()


__all__ = ['HDF5DataSet']