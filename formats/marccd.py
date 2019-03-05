import os
import math
import struct
import re
import numpy
from PIL import Image
from ..import utils
from . import DataSet


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
    header['filename'] = filename

    det_mm = int(round(header['pixel_size'] * header['detector_size'][0]))
    header['detector_type'] = 'mar%d' % det_mm
    header['format'] = 'TIFF'

    if with_image:
        raw_img = Image.open(filename)
        data = numpy.fromstring(raw_img.tobytes(), 'H').reshape(*header['detector_size']).transpose()
        image = raw_img.convert('I')
    else:
        data = None
        image = None

    return header, data, image


class MarCCDDataSet(DataSet):
    def __init__(self, filename, header_only=False):
        super(MarCCDDataSet, self).__init__()
        self.filename = filename
        self.header = {}
        p0 = re.compile('^(?P<root_name>.+)_\d+\.[^.]+$')
        m0 = p0.match(self.filename)
        if m0:
            params = m0.groupdict()
            self.root_name = params['root_name']
        else:
            self.root_name = filename
        self.name = os.path.basename(self.root_name)
        self.current_frame = 1
        self.raw_header, self.raw_data, self.raw_image = read_marccd(filename)
        self.read_header()
        if not header_only:
            self.read_image()

    def read_header(self):
        self.header = {}
        self.header.update(self.raw_header)
        self.header.update({
            'name': self.name,
            'format': 'TIFF',
            'dataset': utils.file_sequences(self.filename),
        })
        if self.header['dataset']:
            self.current_frame = self.header['dataset']['current']

    def read_image(self):
        self.image = self.raw_image
        self.data = self.raw_data
        self.header['average_intensity'] = max(0.0, self.data.mean())
        self.header['min_intensity'], self.header['max_intensity'] = self.data.min(), self.data.max()
        self.header['gamma'] = utils.calc_gamma(self.header['average_intensity'])

    def check_disk_frames(self):
        self.header['dataset'] = utils.file_sequences(self.filename)

    def get_frame(self, index=1):
        """
        Load a specific frame
        :param index: frame index
        :return:
        """
        if self.header['dataset']:
            tmpl = self.header['dataset']['name'].format(index)
            filename = tmpl.format(index)
            if os.path.exists(filename):
                self.raw_header, self.raw_data, self.raw_image = read_marccd(filename, True)
                self.read_header()
                self.read_image()
                self.current_frame = index
                return True
        return False

    def next_frame(self):
        """Load the next frame in the dataset"""
        next_frame = self.current_frame + 1
        return self.get_frame(next_frame)

    def prev_frame(self):
        """Load the previous frame in the dataset"""
        next_frame = self.current_frame - 1
        return self.get_frame(next_frame)



__all__ = ['MarCCDDataSet']
