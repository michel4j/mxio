import itertools
import os
import re
from datetime import datetime

import cv2
import hdf5plugin
import iso8601
import h5py
import numpy
import pytz

from . import DataSet
from .. import utils
from ..log import get_module_logger

# Configure Logging
logger = get_module_logger('imageio')

HEADER_FIELDS = {
    'detector_type': '/entry/instrument/detector/description',
    'two_theta': '/entry/instrument/detector/goniometer/two_theta_range_average',
    'pixel_size': '/entry/instrument/detector/x_pixel_size',
    'exposure_time': '/entry/instrument/detector/frame_time',
    'wavelength': '/entry/instrument/beam/incident_wavelength',
    'date': '/entry/instrument/detector/detectorSpecific/data_collection_date',
    'distance': '/entry/instrument/detector/detector_distance',
    'beam_center': ('/entry/instrument/detector/beam_center_x', '/entry/instrument/detector/beam_center_y'),
    'saturated_value': '/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff',
    #'num_frames': '/entry/instrument/detector/detectorSpecific/nimages',
    'energy': '/entry/instrument/detector/detectorSpecific/photon_energy',
    'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
    'detector_size': ('/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                      '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'),

}


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
    'date': convert_date,
    'two_theta': float,
    'pixel_size': lambda v: float(v)*1000,
    'exposure_time': float,
    'wavelength': float,
    'distance': lambda v: float(v)*1000,
    'beam_center': float,
    'saturated_value': int,
    'num_frames': int,
    'energy': float,
    'sensor_thickness': lambda v: float(v)*1000,
    'detector_size': int,
}

OSCILLATION_FIELDS = '/entry/sample/goniometer/{}'
NUMBER_FORMATS = {
    'uint16': numpy.int16,
    'uint32': numpy.int32,
}


class HDF5DataSet(DataSet):
    name = 'Hierarchical Data Format (version 5) data'

    def __init__(self, path, header_only=False):
        super(HDF5DataSet, self).__init__()
        directory, filename = os.path.split(path)

        self.directory = directory
        self.current_section = None
        self.current_frame = 1
        self.disk_sections = []
        self.section_names = []

        p0 = re.compile('^(?P<root_name>.+)_master\.h5$')
        p1 = re.compile('^(?P<root_name>.+)_(?P<section>data_\d+)\.h5')
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
            self.current_section = params['section']
        else:
            self.master_file = path
            self.root_name = os.path.splitext(filename)[0]
        self.name = self.root_name
        self.raw = h5py.File(self.master_file, 'r')
        self.mask = None
        self.data = None
        self.read_dataset()

    def read_dataset(self):
        self.header = {}
        for key, field in HEADER_FIELDS.items():
            converter = CONVERTERS.get(key, lambda v: v)
            try:
                if not isinstance(field, (tuple, list)):
                    self.header[key] = converter(self.raw[field][()])
                else:
                    self.header[key] = tuple(converter(self.raw[sub_field][()]) for sub_field in field)
            except ValueError:
                logger.error('Field corresponding to {} not found!'.format(key))

        self.header['name'] = self.name
        self.header['format'] = 'HDF5'
        self.header['filename'] = os.path.basename(self.master_file)
        self.sections = {
            name: (d.attrs['image_nr_low'], d.attrs['image_nr_high']) for name, d in self.raw['/entry/data'].items()
        }

        self.section_names = sorted(self.sections.keys())
        if not self.current_section:
            self.current_section = self.section_names[0]
        self.current_frame = self.sections[self.current_section][0]

        # try to find oscillation axis and parameters as first non-zero average
        for axis in ['chi', 'kappa', 'omega', 'phi']:
            start_angles = self.raw[OSCILLATION_FIELDS.format(axis)][()]
            delta_angle = self.raw[OSCILLATION_FIELDS.format(axis + '_range_average')][()]
            total_angle = self.raw[OSCILLATION_FIELDS.format(axis + '_range_total')][()]
            self.header['start_angle'] = start_angles[0]
            self.header['delta_angle'] = delta_angle
            self.header['total_angle'] = total_angle
            self.header['rotation_axis'] = axis
            self.start_angles = start_angles
            if start_angles.mean() != 0.0 and delta_angle*total_angle != 0.0:
                break

        self.check_disk_sections()
        self.read_image()

    def read_image(self):
        if self.mask is None:
            self.mask = numpy.invert(
                self.raw['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].view(bool)
            )

        section = self.raw['/entry/data/{}'.format(self.current_section)]
        frame_index = max(self.current_frame - self.sections[self.current_section][0], 0)
        raw_data = section[frame_index]
        data_type = NUMBER_FORMATS[str(raw_data.dtype)]
        data = raw_data.view(data_type)

        # use a  quarter of the image as a subset of data or fast analysis
        stats_subset = data[:data.shape[0]//2, :data.shape[1]//2]
        valid = self.mask[:data.shape[0]//2, :data.shape[1]//2]
        self.stats_data = stats_subset[valid]

        self.header['average_intensity'], self.header['std_dev'] = numpy.ravel(cv2.meanStdDev(self.stats_data))
        self.header['min_intensity'] = 0
        self.header['max_intensity'] = float(self.stats_data.max())  # Estimate
        self.header['overloads'] = 4*(self.stats_data == self.header['saturated_value']).sum()  # Fast estimate
        self.header['frame_number'] = self.current_frame
        self.data = data

    def check_disk_sections(self):
        data_file = os.path.join(self.directory, self.root_name + '_' + self.section_names[0] + '.h5')
        dataset = utils.file_sequences(data_file)
        if dataset:
            link_tmpl = dataset['name']
            self.disk_sections = {
                section_name: images
                for i, (section_name, images) in enumerate(self.sections.items())
                if os.path.exists(os.path.join(self.directory, link_tmpl.format(i+1)))
            }
            frames = list(itertools.chain.from_iterable(range(v[0], v[1] + 1) for v in self.disk_sections.values()))
            width = 6
            template = '{root_name}_{{field}}.h5'.format(root_name=self.root_name)
            regex = r'^{root_name}_data_(\d{{{width}}}).h5$'.format(width=width, root_name=self.root_name)
            current = self.current_frame

            self.header['dataset'] = {
                'name': '{root_name}_master.h5'.format(root_name=self.root_name),
                'start_time': self.header['date'].replace(tzinfo=pytz.utc),
                'label': self.root_name,
                'directory': self.directory,
                'template': template.format(field='?' * width),
                'reference': dataset['reference'],
                'regex': regex,
                'start_angle': float(self.start_angles[0]),
                'sequence': frames,
                'current': current
            }
        else:
            self.header['dataset'] = {}

    def get_frame(self, index=1):
        """
        Load a specific frame
        :param index: frame index
        :return:
        """
        self.check_disk_sections()
        for section_name, section_limits in self.sections.items():
            if section_name in self.disk_sections:
                if section_limits[0] <= index <= section_limits[1]:
                    self.current_frame = index
                    self.current_section = section_name
                    self.header['start_angle'] = self.start_angles[self.current_frame-1]
                    self.read_image()
                    return True
        return False

    def next_frame(self):
        """Load the next frame in the dataset"""
        self.check_disk_sections()
        next_frame = self.current_frame + 1
        section_limits = self.sections[self.current_section]
        if next_frame <= section_limits[1]:
            self.current_frame = next_frame
        else:
            # skip to first image of next section
            i = self.section_names.index(self.current_section) + 1
            if i <= len(self.section_names)-1:
                next_section = self.section_names[i]
                next_frame = self.sections[next_section][0]
                if next_section in self.disk_sections:
                    self.current_frame = next_frame
                    self.current_section = next_section
                else:
                    return False
            else:
                return False
        self.header['start_angle'] = float(self.start_angles[self.current_frame - 1])
        self.read_image()
        return True

    def prev_frame(self):
        """Load the previous frame in the dataset"""
        self.check_disk_sections()
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
                if next_section in self.disk_sections:
                    self.current_frame = next_frame
                    self.current_section = next_section
                else:
                    return False
            else:
                return False
        self.header['start_angle'] = self.start_angles[self.current_frame - 1]
        self.read_image()
        return True


__all__ = ['HDF5DataSet']
