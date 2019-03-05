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
        p0 = re.compile('^(?P<root_name>.+)_master\.h5$')
        p1 = re.compile('^(?P<root_name>.+)_data_\d+\.h5')
        m0 = p0.match(filename)
        m1 = p1.match(filename)
        if m0:
            params = m0.groupdict()
            self.master_file = filename
            self.root_name = params['root_name']
        elif m1:
            params = m1.groupdict()
            self.master_file = params['root_name'] + '_master.h5'
            self.root_name = params['root_name']
        else:
            self.master_file = filename
            self.root_name = ""
            logger.error('Unable to recognize HDF5 dataset')
        self.directory, self.filename = os.path.split(os.path.abspath(self.master_file))
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

        self.header['file_format'] ='HDF5'
        self.header['distance'] *= 1000
        self.header['sensor_thickness'] *= 1000
        self.header['pixel_size'] = 1000*self.header['pixel_size'][0]
        self.header['filename'] = self.master_file
        self.header['sections'] = sorted(self.raw['/entry/data'].keys())
        self.header['dataset'] = utils.file_sequences(self.root_name + '_' + self.header['sections'][0] + '.h5')

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

    def read_image(self):
        if self.mask is None:
            self.mask = numpy.invert(self.raw['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool))

        section = self.raw['/entry/data/{}'.format(self.header['sections'][0])]
        data = section[0]
        valid = self.mask & (data < self.header['saturated_value'])
        self.header['average_intensity'] = data[valid].mean()
        self.header['min_intensity'] = 0
        self.header['max_intensity'] = data[valid].max()
        self.header['gamma'] = utils.calc_gamma(self.header['average_intensity'])
        self.header['overloads'] = self.mask.sum() - valid.sum()
        self.image = Image.fromarray(data)
        self.image = self.image.convert('I')
        self.data = data.T


__all__ = ['HDF5DataSet']
