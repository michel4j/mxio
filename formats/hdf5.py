import bitshuffle.h5
import os
import re
import h5py
import numpy

from PIL import Image
from ..log import get_module_logger
from .. import utils
from . import DataSet

# Configure Logging
logger = get_module_logger('imageio')

HEADER_FIELDS = {
    'detector_type': '/entry/instrument/detector/description',
    'two_theta': '/entry/instrument/detector/goniometer/two_theta_range_average',
    'pixel_size': ('/entry/instrument/detector/x_pixel_size',
                   '/entry/instrument/detector/y_pixel_size'),
    'exposure_time': '/entry/instrument/detector/frame_time',
    'wavelength': '/entry/instrument/beam/incident_wavelength',
    'date': '/entry/instrument/detector/detectorSpecific/data_collection_date',
    'distance': '/entry/instrument/detector/detector_distance',
    'beam_center': ('/entry/instrument/detector/beam_center_x', '/entry/instrument/detector/beam_center_y'),
    'saturated_value': '/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff',
    'num_frames': '/entry/instrument/detector/detectorSpecific/nimages',
    'energy': '/entry/instrument/detector/detectorSpecific/photon_energy',
    'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
    'detector_size': ('/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                      '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'),

}

OSCILLATION_FIELDS = '/entry/sample/goniometer/{}'


class HDF5DataSet(DataSet):
    def __init__(self, filename, header_only=False):
        super(HDF5DataSet, self).__init__()
        self.master_file = filename
        p0 = re.compile('^(?P<root_name>.+)_master\.h5$')
        p1 = re.compile('^(?P<root_name>.+)_(?P<section>data_\d+)\.h5')
        m0 = p0.match(filename)
        m1 = p1.match(filename)
        self.current_section = None
        self.current_frame = 1
        self.disk_sections = []
        self.section_names = []
        if m0:
            params = m0.groupdict()
            self.root_name = params['root_name']
        elif m1:
            params = m1.groupdict()
            self.master_file = params['root_name'] + '_master.h5'
            self.root_name = params['root_name']
            self.current_section = params['section']
        else:
            self.master_file = filename
            self.root_name = os.path.splitext(self.master_file)[0]
        self.directory = os.path.dirname(os.path.abspath(self.master_file))
        self.name = os.path.basename(self.root_name)
        self.raw = h5py.File(self.master_file, 'r')
        self.mask = None
        self.read_header()

        if not header_only:
            self.read_image()

    def read_header(self):
        self.header = {}
        for key, field in HEADER_FIELDS.items():
            try:
                if not isinstance(field, (tuple, list)):
                    self.header[key] = self.raw[field].value
                else:
                    self.header[key] = tuple(self.raw[sub_field].value for sub_field in field)
            except ValueError:
                logger.error('Field corresponding to {} not found!'.format(key))

        self.header['name'] = self.name
        self.header['format'] ='HDF5'
        self.header['distance'] *= 1000
        self.header['sensor_thickness'] *= 1000
        self.header['pixel_size'] = 1000*self.header['pixel_size'][0]
        self.header['filename'] = self.master_file
        self.header['sections'] = {
            name: (d.attrs['image_nr_low'], d.attrs['image_nr_high']) for name, d in self.raw['/entry/data'].items()
        }
        self.section_names = sorted(self.header['sections'].keys())
        if not self.current_section:
            self.current_section = self.section_names[0]
        self.current_frame = self.header['sections'][self.current_section][0]

        # try to find oscillation axis and parameters as first non-zero average
        for axis in ['chi', 'kappa', 'omega', 'phi']:
            start_angles = self.raw[OSCILLATION_FIELDS.format(axis)].value
            delta_angle = self.raw[OSCILLATION_FIELDS.format(axis + '_range_average')].value
            total_angle = self.raw[OSCILLATION_FIELDS.format(axis + '_range_total')].value
            self.header['start_angle'] = start_angles[0]
            self.header['delta_angle'] = delta_angle
            self.header['total_angle'] = total_angle
            self.header['rotation_axis'] = axis
            if start_angles.mean() != 0.0 and delta_angle*total_angle != 0.0:
                break

        self.check_disk_sections()

    def read_image(self):
        if self.mask is None:
            self.mask = numpy.invert(
                self.raw['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool)
            )

        section = self.raw['/entry/data/{}'.format(self.current_section)]
        frame_index = max(self.current_frame - self.header['sections'][self.current_section][0], 0)
        data = section[frame_index]
        valid = self.mask & (data < self.header['saturated_value'])
        self.header['average_intensity'] = data[valid].mean()
        self.header['min_intensity'] = 0
        self.header['max_intensity'] = data[valid].max()
        self.header['gamma'] = utils.calc_gamma(self.header['average_intensity'])
        self.header['overloads'] = self.mask.sum() - valid.sum()
        self.image = Image.fromarray(data)
        self.image = self.image.convert('I')
        self.data = data.T

    def check_disk_sections(self):
        self.header['dataset'] = utils.file_sequences(self.root_name + '_' + self.section_names[0] + '.h5')
        if self.header['dataset']:
            tmpl = self.header['dataset']['name']
            self.disk_sections = [
                section_name for i, section_name in enumerate(self.section_names)
                if os.path.exists(tmpl.format(i+1))
            ]

    def get_frame(self, index=1):
        """
        Load a specific frame
        :param index: frame index
        :return:
        """
        self.check_disk_sections()
        for section_name, section_limits in self.header['sections'].items():
            if section_name in self.disk_sections:
                if section_limits[0] <= index <= section_limits[1]:
                    self.current_frame = index
                    self.current_section = section_name
                    self.read_image()
                    return True
        return False

    def next_frame(self):
        """Load the next frame in the dataset"""
        self.check_disk_sections()
        next_frame = self.current_frame + 1
        section_limits = self.header['sections'][self.current_section]
        if next_frame <= section_limits[1]:
            self.current_frame = next_frame
        else:
            # skip to first image of next section
            i = self.section_names.index(self.current_section) + 1
            if i <= len(self.section_names)-1:
                next_section = self.section_names[i]
                next_frame = self.header['sections'][next_section][0]
                if next_section in self.disk_sections:
                    self.current_frame = next_frame
                    self.current_section = next_section
                else:
                    return False
            else:
                return False
        self.read_image()
        return True

    def prev_frame(self):
        """Load the previous frame in the dataset"""
        self.check_disk_sections()
        next_frame = self.current_frame - 1
        section_limits = self.header['sections'][self.current_section]
        if next_frame >= section_limits[0]:
            self.current_frame = next_frame
        else:
            # skip to last image of prev section
            i = self.section_names.index(self.current_section) - 1
            if i >= 0:
                next_section = self.section_names[i]
                next_frame = self.header['sections'][next_section][1]
                if next_section in self.disk_sections:
                    self.current_frame = next_frame
                    self.current_section = next_section
                else:
                    return False
            else:
                return False
        self.read_image()
        return True


__all__ = ['HDF5DataSet']
