import os
import math
import struct
import cv2

from ..import utils
from . import DataSet
from ..log import get_module_logger

# Configure Logging
logger = get_module_logger('mxio')


def read_marccd(filename, with_image=True):
    header = {}

    # Read MarCCD header
    header_format = 'I16s39I80x'  # 256 bytes
    statistics_format = '3Q7I9I40x128H'  # 128 + 256 bytes
    goniostat_format = '28i16x'  # 128 bytes
    detector_format = '5i9i9i9i'  # 128 bytes
    source_format = '10i16x10i32x'  # 128 bytes
    file_format = '128s128s64s32s32s32s512s96x'  # 1024 bytes
    dataset_format = '512s'  # 512 bytes
    # image_format = '9437184H'

    marccd_header_format = header_format + statistics_format
    marccd_header_format += goniostat_format + detector_format + source_format
    marccd_header_format += file_format + dataset_format + '512x'
    myfile = open(filename, 'rb')

    tiff_header = myfile.read(1024)
    del tiff_header
    header_pars = struct.unpack(header_format, myfile.read(256))
    statistics_pars = struct.unpack(statistics_format, myfile.read(128 + 256))
    goniostat_pars = struct.unpack(goniostat_format, myfile.read(128))
    detector_pars = struct.unpack(detector_format, myfile.read(128))
    source_pars = struct.unpack(source_format, myfile.read(128))
    myfile.close()

    # extract some values from the header
    # use image center if detector origin is (0,0)
    if goniostat_pars[1] / 1e3 + goniostat_pars[2] / 1e3 < 0.1:
        header['beam_center'] = header_pars[17] / 2.0, header_pars[18] / 2.0
    else:
        header['beam_center'] = goniostat_pars[1] / 1e3, goniostat_pars[2] / 1e3

    header['distance'] = goniostat_pars[0] / 1e3
    header['wavelength'] = source_pars[3] / 1e5
    header['pixel_size'] = detector_pars[1] / 1e6
    header['delta_angle'] = goniostat_pars[24] / 1e3
    header['start_angle'] = goniostat_pars[(7 + goniostat_pars[23])] / 1e3
    header['exposure_time'] = goniostat_pars[4] / 1e3
    header['min_intensity'] = statistics_pars[3]
    header['max_intensity'] = statistics_pars[4]
    header['rms_intensity'] = statistics_pars[6] / 1e3
    header['average_intensity'] = statistics_pars[5] / 1e3
    header['overloads'] = statistics_pars[8]
    header['saturated_value'] = header_pars[23]
    header['two_theta'] = (goniostat_pars[7] / 1e3) * math.pi / -180.0
    header['detector_size'] = (header_pars[17], header_pars[18])
    header['filename'] = os.path.basename(filename)

    det_mm = int(round(header['pixel_size'] * header['detector_size'][0]))
    header['detector_type'] = 'mar%d' % det_mm
    header['format'] = 'TIFF'

    data = cv2.imread(filename, -1)
    return header, data


class MarCCDDataSet(DataSet):
    name = 'marCCD Area Detector Image'

    def __init__(self, filename, header_only=False):
        super(MarCCDDataSet, self).__init__()
        self.filename = filename
        self.header = {}
        self.name = os.path.splitext(os.path.basename(self.filename))[0]
        self.current_frame = 1
        self.raw_header, self.raw_data = read_marccd(filename)
        self.read_dataset()

    def read_dataset(self):
        self.header = {}
        self.header.update(self.raw_header)
        self.header.update({
            'format': 'TIFF',
            'dataset': utils.file_sequences(self.filename),
        })
        self.current_frame = self.header['dataset']['current']
        self.header['name'] = self.header['dataset']['label']
        self.header['dataset'].update({
            'start_angle': (
                    self.header['start_angle'] - self.header['delta_angle'] * (self.header['dataset']['current'] - 1)
            )
        })

        self.data = self.raw_data
        stats_subset = self.data[:self.data.shape[0] // 2, :self.data.shape[1] // 2]
        valid = (stats_subset > 0) & (stats_subset <= self.header['saturated_value'])
        self.stats_data = stats_subset[valid]
        self.header['std_dev'] = self.stats_data.std()
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
                self.raw_header, self.raw_data = read_marccd(filename)
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



__all__ = ['MarCCDDataSet']
