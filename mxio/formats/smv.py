
import ctypes
import re
import cv2
import os
import numpy

from ..import utils
from . import DataSet
from ..log import get_module_logger

# Configure Logging
logger = get_module_logger('mxio')

DECODER_DICT = {
    "unsigned_short": (ctypes.c_uint16, 'F;16','F;16B'),
    "unsigned_int": (ctypes.c_uint, 'F;32','F;32B'),
    "signed_short": (ctypes.c_int16, 'F;16S','F;16BS'),
    "signed_int": (ctypes.c_int, 'F;32S','F;32BS'),
}

def read_smv(filename, with_image=True):
    info = {}
    myfile = open(filename, 'r')
    raw = myfile.read(512)
    raw_entries = raw.split('\n')
    tmp_info = {}
    epat = re.compile('^(?P<key>[\w]+)=(?P<value>.+);')
    for line in raw_entries:
        m = epat.match(line)
        if m:
            tmp_info[m.group('key').lower()] = m.group('value').strip()
    # Read remaining header if any
    _header_size = int(tmp_info['header_bytes'])
    if _header_size > 512:
        raw = myfile.read(_header_size - 512)
        raw_entries = raw.split('\n')
        for line in raw_entries:
            m = epat.match(line)
            if m:
                tmp_info[m.group('key').lower()] = m.group('value').strip()
    myfile.close()
    _type = tmp_info.get('type', "unsigned_short")
    _el_type = DECODER_DICT[_type][0]

    # decoder suffix for endianess
    # if tmp_info.get('byte_order') == 'big_endian':
    #     _raw_decoder = DECODER_DICT[_type][2]
    # else:
    #     _raw_decoder = DECODER_DICT[_type][1]

    info['delta_angle'] = float(tmp_info['osc_range'])
    info['distance'] = float(tmp_info['distance'])
    info['wavelength'] = float(tmp_info['wavelength'])
    info['exposure_time'] = float(tmp_info['time'])
    info['pixel_size'] = float(tmp_info['pixel_size'])
    orgx = float(tmp_info['beam_center_x']) / info['pixel_size']
    orgy = float(tmp_info['beam_center_y']) / info['pixel_size']
    info['detector_size'] = (int(tmp_info['size1']), int(tmp_info['size2']))
    info['beam_center'] = (orgx, info['detector_size'][1] - orgy)

    # use image center if detector origin is (0,0)
    if sum(info['beam_center']) < 0.1:
        info['beam_center'] = (info['detector_size'][0] / 2.0, info['detector_size'][1] / 2.0)
    info['start_angle'] = float(tmp_info['osc_start'])
    if tmp_info.get('twotheta') is not None:
        info['two_theta'] = float(tmp_info['twotheta'])
    else:
        info['two_theta'] = 0.0

    if info['detector_size'][0] == 2304:
        info['detector_type'] = 'q4'
    elif info['detector_size'][0] == 1152:
        info['detector_type'] = 'q4-2x'
    elif info['detector_size'][0] == 4096:
        info['detector_type'] = 'q210'
    elif info['detector_size'][0] == 2048:
        info['detector_type'] = 'q210-2x'
    elif info['detector_size'][0] == 6144:
        info['detector_type'] = 'q315'
    elif info['detector_size'][0] == 3072:
        info['detector_type'] = 'q315-2x'
    info['filename'] = os.path.basename(filename)
    info['saturated_value'] = 2 ** (8 * ctypes.sizeof(_el_type)) - 1


    num_el = info['detector_size'][0] * info['detector_size'][1]
    el_size = ctypes.sizeof(_el_type)
    data_size = num_el * el_size
    with open(filename, 'rb') as myfile:
        myfile.read(_header_size)
        data = myfile.read(data_size)
    raw_data = numpy.fromstring(data, dtype=_el_type).reshape(*info['detector_size'])

    return info, raw_data


class SMVDataSet(DataSet):
    name = 'SMV Area Detector Image'
    
    def __init__(self, filename, header_only=False):
        super(SMVDataSet, self).__init__()
        self.filename = filename
        self.header = {}
        self.data = None
        self.image = None
        self.name = os.path.splitext(os.path.basename(self.filename))[0]
        self.current_frame = 1
        self.raw_header, self.raw_data = read_smv(filename)
        self.read_dataset()

    def read_dataset(self):
        self.header = {}
        self.header.update(self.raw_header)
        self.header.update({
            'name': self.name,
            'format': 'SMV',
            'dataset': utils.file_sequences(self.filename),
        })
        if self.header['dataset']:
            self.current_frame = self.header['dataset']['current']
            self.header['name'] = self.header['dataset']['label']
            self.header['dataset'].update({
                'start_angle': (
                    self.header['start_angle'] - self.header['delta_angle'] * ( self.header['dataset']['current'] - 1)
                )
            })

        self.data = self.raw_data
        stats_subset = self.data[:self.data.shape[0] // 2, :self.data.shape[1] // 2]
        valid = (stats_subset > 0) & (stats_subset <= self.header['saturated_value'])
        self.stats_data = stats_subset[valid]

        self.header['average_intensity'], self.header['std_dev'] = numpy.ravel(cv2.meanStdDev(self.stats_data))
        self.header['min_intensity'] = self.stats_data.min()
        self.header['max_intensity'] = self.stats_data.max()
        self.header['overloads'] = 4*(self.stats_data == self.header['saturated_value']).sum()
        self.header['frame_number'] = self.current_frame

    def check_disk_frames(self):
        self.header['dataset'] = utils.file_sequences(self.filename)

    def get_frame(self, index=1):
        """
        Load a specific frame
        :param index: frame index
        :return:
        """
        if self.header['dataset']:
            filename = os.path.join(
                self.header['dataset']['directory'],
                self.header['dataset']['name'].format(index),
            )
            if os.path.exists(filename):
                self.filename = filename
                self.raw_header, self.raw_data = read_smv(filename)
                self.current_frame = index
                self.read_dataset()
                return True
        return False

    def next_frame(self):
        """Load the next frame in the dataset"""
        self.check_disk_frames()
        if self.header['dataset']:
            next_pos = self.header['dataset']['sequence'].index(self.current_frame) + 1
            if next_pos < len(self.header['dataset']['sequence']):
                next_frame = self.header['dataset']['sequence'][next_pos]
                return self.get_frame(next_frame)
        return False

    def prev_frame(self):
        """Load the previous frame in the dataset"""
        self.check_disk_frames()
        if self.header['dataset']:
            prev_pos = self.header['dataset']['sequence'].index(self.current_frame) - 1
            if prev_pos >= 0:
                prev_frame = self.header['dataset']['sequence'][prev_pos]
                return self.get_frame(prev_frame)
        return False



__all__ = ['SMVDataSet']
