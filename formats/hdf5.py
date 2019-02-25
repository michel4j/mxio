import bitshuffle.h5
import tables
import h5py
import numpy

from PIL import Image
from ..log import get_module_logger
from .. import utils

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
    #'start_angle': '/entry/sample/goniometer/omega',
    #'delta_angle': '/entry/sample/goniometer/omega_range_average',
    'saturated_value': '/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff',
    'num_frames': '/entry/instrument/detector/detectorSpecific/nimages',
    'energy': '/entry/instrument/detector/detectorSpecific/photon_energy',
    'sensor_thickness': '/entry/instrument/detector/sensor_thickness',
    'detector_size': ('/entry/instrument/detector/detectorSpecific/x_pixels_in_detector',
                      '/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'),

}

OSCILLATION_FIELDS = '/entry/sample/goniometer/{}'


class HDF5DataFile(object):
    def __init__(self, filename, header_only=False):
        self.raw = h5py.File(filename, 'r')
        self.filename = filename
        self._read_header()
        if not header_only:
            self._read_frame()

    def _read_header(self):
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
        self.header['pixel_size'] = tuple(1000*x for x in self.header['pixel_size'])
        self.header['filename'] = self.filename

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


    def _read_frame(self):
        keys = sorted(self.raw['/entry/data'].keys())
        section = self.raw['/entry/data/{}'.format(keys[0])]
        'Section has {} frames'.format(section.shape[0])
        data = section[50]
        self.header['average_intensity'] = max(0.0, data.mean())
        self.header['min_intensity'] = 0 #data.min()
        self.header['gamma'] = utils.calc_gamma(self.header['average_intensity'])
        self.header['overloads'] = 0 #en(numpy.where(data >= self.header['saturated_value'])[0])
        self.data = data
        self.image = Image.fromarray(data, 'F')
        self.image = self.image.convert('I')




__all__ = ['HDF5DataFile']
