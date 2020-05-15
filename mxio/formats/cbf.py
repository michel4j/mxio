import os
import re
from ctypes import *
import cv2
import numpy

from . import DataSet
from .. import utils, parser
from ..common import *
from ..log import get_module_logger

# Configure Logging
logger = get_module_logger('imageio')

# Define CBF Error Code constants
CBF_FORMAT = 0x00000001  # 1
CBF_ALLOC = 0x00000002  # 2
CBF_ARGUMENT = 0x00000004  # 4
CBF_ASCII = 0x00000008  # 8
CBF_BINARY = 0x00000010  # 16
CBF_BITCOUNT = 0x00000020  # 32
CBF_ENDOFDATA = 0x00000040  # 64
CBF_FILECLOSE = 0x00000080  # 128
CBF_FILEOPEN = 0x00000100  # 256
CBF_FILEREAD = 0x00000200  # 512
CBF_FILESEEK = 0x00000400  # 1024
CBF_FILETELL = 0x00000800  # 2048
CBF_FILEWRITE = 0x00001000  # 4096
CBF_IDENTICAL = 0x00002000  # 8192
CBF_NOTFOUND = 0x00004000  # 16384
CBF_OVERFLOW = 0x00008000  # 32768
CBF_UNDEFINED = 0x00010000  # 65536
CBF_NOTIMPLEMENTED = 0x00020000  # 131072
CBF_NOCOMPRESSION = 0x00040000  # 262144

PLAIN_HEADERS = 0x0001  # Use plain ASCII headers
MIME_HEADERS = 0x0002  # Use MIME headers
MSG_NODIGEST = 0x0004  # Do not check message digests
MSG_DIGEST = 0x0008  # Check message digests
MSG_DIGESTNOW = 0x0010  # Check message digests immediately
MSG_DIGESTWARN = 0x0020  # Warn on message digests immediately
PAD_1K = 0x0020  # Pad binaries with 1023 0's
PAD_2K = 0x0040  # Pad binaries with 2047 0's
PAD_4K = 0x0080  # Pad binaries with 4095 0's

CBF_ERROR_MESSAGES = {
    CBF_FORMAT: 'Invalid File Format',
    CBF_ALLOC: 'Memory Allocation Error',
    CBF_ARGUMENT: 'Invalid function arguments',
    CBF_ASCII: 'Value is ASCII (not binary)',
    CBF_BINARY: 'Value is binary (not ASCII)',
    CBF_BITCOUNT: 'Expected number of bits does not match actual number written',
    CBF_ENDOFDATA: 'End of data was reached before end of array',
    CBF_FILECLOSE: 'File close error',
    CBF_FILEOPEN: 'File open error',
    CBF_FILEREAD: 'File read error',
    CBF_FILESEEK: 'File seek error',
    CBF_FILETELL: 'File tell error',
    CBF_FILEWRITE: 'File write error',
    CBF_IDENTICAL: 'Data block with identical name already exists',
    CBF_NOTFOUND: 'Data block/category/column/row does not exist',
    CBF_OVERFLOW: 'Value overflow error. The value has been truncated',
    CBF_UNDEFINED: 'Requested number is undefined',
    CBF_NOTIMPLEMENTED: 'Requested functionality is not implemented',
    CBF_NOCOMPRESSION: 'No compression',
}

DECODER_DICT = {
    "unsigned 16-bit integer": (c_uint16, 'F;16', 'F;16B'),
    "unsigned 32-bit integer": (c_uint32, 'F;32', 'F;32B'),
    "signed 16-bit integer": (c_int16, 'F;16S', 'F;16BS'),
    "signed 32-bit integer": (c_int32, 'F;32S', 'F;32BS'),
}

ELEMENT_TYPES = {
    "signed 32-bit integer": c_int32,
}


def _format_error(code):
    errors = []
    for k, v in CBF_ERROR_MESSAGES.items():
        if (code | k) == code:
            errors.append(v)
    return ', '.join(errors)


try:
    cbflib = cdll.LoadLibrary('libcbf.so.0')
except Exception:
    raise TypeError('CBF Shared Library not available')


try:
    libc = cdll.LoadLibrary('libc.so.6')
except Exception:
    raise TypeError('C Shared Library not available')

# define argument and return types
libc.fopen.argtypes = [c_char_p, c_char_p]
libc.fopen.restype = c_void_p

cbflib.cbf_make_handle.argtypes = [c_void_p]
cbflib.cbf_free_handle.argtypes = [c_void_p]
cbflib.cbf_read_file.argtypes = [c_void_p, c_void_p, c_int]
cbflib.cbf_read_widefile.argtypes = [c_void_p, c_void_p, c_int]
cbflib.cbf_get_wavelength.argtypes = [c_void_p, POINTER(c_double)]
cbflib.cbf_get_integration_time.argtypes = [c_void_p, c_uint, POINTER(c_double)]
cbflib.cbf_get_image_size.argtypes = [c_void_p, c_uint, c_uint, POINTER(c_size_t), POINTER(c_size_t)]
cbflib.cbf_construct_goniometer.argtypes = [c_void_p, c_void_p]
cbflib.cbf_construct_detector.argtypes = [c_void_p, c_void_p, c_uint]
cbflib.cbf_construct_reference_detector.argtypes = [c_void_p, c_void_p, c_uint]
cbflib.cbf_require_reference_detector.argtypes = [c_void_p, c_void_p, c_uint]
cbflib.cbf_read_template.argtypes = [c_void_p, c_void_p]
cbflib.cbf_get_pixel_size.argtypes = [c_void_p, c_uint, c_int, POINTER(c_double)]
cbflib.cbf_get_detector_distance.argtypes = [c_void_p, POINTER(c_double)]
cbflib.cbf_get_beam_center.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double),
                                       POINTER(c_double)]
cbflib.cbf_get_rotation_range.argtypes = [c_void_p, c_uint, POINTER(c_double), POINTER(c_double)]
cbflib.cbf_get_detector_normal.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
cbflib.cbf_get_rotation_axis.argtypes = [c_void_p, c_uint, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
cbflib.cbf_get_image.argtypes = [c_void_p, c_uint, c_uint, c_void_p, c_size_t, c_int, c_size_t, c_size_t]
cbflib.cbf_get_detector_id.argtypes = [c_void_p, c_uint, c_void_p]
cbflib.cbf_parse_mimeheader.argtypes = [c_void_p, POINTER(c_int), POINTER(c_size_t), POINTER(c_long),
                                        c_void_p, POINTER(c_uint), POINTER(c_int), POINTER(c_int), POINTER(c_int),
                                        c_void_p,
                                        POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t),
                                        POINTER(c_size_t)]
cbflib.cbf_get_integerarray.argtypes = [c_void_p, POINTER(c_int), c_void_p, c_size_t, c_int, c_size_t,
                                        POINTER(c_size_t)]
cbflib.cbf_select_datablock.argtypes = [c_void_p, c_uint]
cbflib.cbf_count_datablocks.argtypes = [c_void_p, POINTER(c_uint)]
cbflib.cbf_find_datablock.argtypes = [c_void_p, c_char_p]
cbflib.cbf_find_category.argtypes = [c_void_p, c_char_p]
cbflib.cbf_find_column.argtypes = [c_void_p, c_char_p]
cbflib.cbf_datablock_name.argtypes = [c_void_p, c_void_p]
cbflib.cbf_get_overload.argtypes = [c_void_p, c_uint, POINTER(c_double)]


def get_max_int(t):
    v = t(2 ** (8 * sizeof(t)) - 1).value
    if v == -1:  # signed
        return c_double(2 ** (8 * sizeof(t) - 1) - 1)
    else:  # unsigned
        return c_double(2 ** (8 * sizeof(t) - 1))


def read_cbf(filename, with_image=True):
    handle = c_void_p()
    goniometer = c_void_p()
    detector = c_void_p()

    # make the handle and read the file
    res = cbflib.cbf_make_handle(byref(handle))
    fp = libc.fopen(filename.encode('utf-8'), b"rb")
    res |= cbflib.cbf_read_template(handle, fp)
    res |= cbflib.cbf_construct_goniometer(handle, byref(goniometer))
    res |= cbflib.cbf_require_reference_detector(handle, byref(detector), 0)

    # read mime
    hr = re.compile(r'^(.+):\s+(.+)$')
    bin_st = re.compile(r'^--CIF-BINARY-FORMAT-SECTION--')
    mime_header = {}
    parse_tokens = {
        "Content-Type": str,
        "Content-Transfer-Encoding": str,
        "Content-MD5": str,
        "X-Binary-Size": int,
        "X-Binary-ID": int,
        "X-Binary-Element-Type": str,
        "X-Binary-Element-Byte-Order": str,
        "X-Binary-Number-of-Elements": int,
        "X-Binary-Size-Fastest-Dimension": int,
        "X-Binary-Size-Second-Dimension": int,
        "X-Binary-Size-Third-Dimension": int,
        "X-Binary-Size-Padding": int
    }

    with open(filename, 'rb') as fh:
        # find start of binary header
        i = 0
        bin_found = False
        while not bin_found and i < 512:
            line = fh.readline().decode()
            bin_found = bin_st.match(line)
            i += 1

        if i >= 512:
            return mime_header

        # extract binary header
        line = fh.readline().decode()
        while line.strip() != '':
            m = hr.match(line)
            if m:
                mime_header[m.group(1)] = parse_tokens[m.group(1)](m.group(2).replace('"', '').strip())
            line = fh.readline().decode()

    # read header
    header = {}
    wvl = c_double(1.0)
    res = cbflib.cbf_get_wavelength(handle, byref(wvl))
    header['wavelength'] = wvl.value

    sz1 = c_size_t(mime_header.get('X-Binary-Size-Fastest-Dimension', 0))
    sz2 = c_size_t(mime_header.get('X-Binary-Size-Second-Dimension', 0))
    header['detector_size'] = (sz1.value, sz2.value)

    px1 = c_double(1.0)
    res |= cbflib.cbf_get_pixel_size(handle, 0, 1, byref(px1))
    header['pixel_size'] = px1.value

    dst = c_double(999.0)
    res |= cbflib.cbf_get_detector_distance(detector, byref(dst))
    header['distance'] = dst.value

    dx, dy = c_double(0.0), c_double(0.0)
    ix, iy = c_double(sz1.value / 2.0), c_double(sz2.value / 2.0)
    res |= cbflib.cbf_get_beam_center(detector, byref(ix), byref(iy), byref(dx), byref(dy))
    header['beam_center'] = (ix.value, iy.value)

    it = c_double(0.0)
    res |= cbflib.cbf_get_integration_time(handle, 0, it)
    header['exposure_time'] = it.value

    st, inc = c_double(0.0), c_double(0.0)
    res |= cbflib.cbf_get_rotation_range(goniometer, 0, byref(st), byref(inc))
    header['start_angle'] = st.value
    header['delta_angle'] = inc.value

    el_type = DECODER_DICT[mime_header.get('X-Binary-Element-Type', 'signed 32-bit integer')][0]
    ovl = get_max_int(el_type)
    res |= cbflib.cbf_get_overload(handle, 0, byref(ovl))
    header['saturated_value'] = ovl.value

    # FIXME Calculate actual two_theta from the beam direction and detector normal
    vx, vy, vz = c_double(0.0), c_double(0.0), c_double(0.0)
    res |= cbflib.cbf_get_detector_normal(detector, byref(vx), byref(vy), byref(vx))
    detector_norm = numpy.array([vx.value, vy.value, vz.value])

    res |= cbflib.cbf_get_rotation_axis(goniometer, 0, byref(vx), byref(vy), byref(vx))
    rot_axis = numpy.array([vx.value, vy.value, vz.value])

    header['two_theta'] = 0.0

    det_id = c_char_p()
    res |= cbflib.cbf_get_detector_id(handle, 0, byref(det_id))
    header['detector_type'] = det_id.value
    if header['detector_type'] is None:
        header['detector_type'] = 'Unknown'
    header['format'] = 'CBF'

    # handle XDS Special cbf files
    if header['distance'] == 999.0 and header['delta_angle'] == 0.0 and header['exposure_time'] == 0.0:
        res = cbflib.cbf_select_datablock(handle, c_uint(0))
        res |= cbflib.cbf_find_category(handle, b"array_data")
        res |= cbflib.cbf_find_column(handle, b"header_convention")
        hdr_type = c_char_p()
        res |= cbflib.cbf_get_value(handle, byref(hdr_type))
        res |= cbflib.cbf_find_column(handle, b"header_contents")
        hdr_contents = c_char_p()
        res |= cbflib.cbf_get_value(handle, byref(hdr_contents))
        if res == 0 and hdr_type.value != 'XDS special':
            logger.debug('miniCBF header type found: {}'.format(hdr_type.value))
            info = parser.parse_text((hdr_contents.value).decode(), (hdr_type.value).decode())
            header['detector_type'] = info['detector'].lower().strip().replace(' ', '')
            header['two_theta'] = 0 if not info['two_theta'] else round(info['two_theta'], 2)
            header['pixel_size'] = round(info['pixel_size'][0] * 1000, 5)
            header['exposure_time'] = info['exposure_time']
            header['wavelength'] = info['wavelength']
            header['distance'] = info['distance'] * 1000
            header['beam_center'] = info['beam_center']
            header['start_angle'] = info['start_angle']
            header['delta_angle'] = info['delta_angle']
            header['saturated_value'] = info['saturated_value']
            header['sensor_thickness'] = info['sensor_thickness'] * 1000
        else:
            logger.debug('miniCBF with no header')
    header['filename'] = os.path.basename(filename)

    num_el = header['detector_size'][0] * header['detector_size'][1]
    el_params = DECODER_DICT[mime_header.get('X-Binary-Element-Type', 'signed 32-bit integer')]
    el_type = el_params[0]
    el_size = sizeof(el_type)
    data = create_string_buffer(num_el * el_size)
    res = cbflib.cbf_get_image(
        handle, 0, 0, byref(data), el_size, 1, header['detector_size'][0], header['detector_size'][1]
    )
    if res != 0:
        # MiniCBF
        res = cbflib.cbf_select_datablock(handle, c_uint(0))
        res |= cbflib.cbf_find_category(handle, b"array_data")
        res |= cbflib.cbf_find_column(handle, b"data")
        binary_id = c_int(mime_header.get('X-Binary-ID', 1))
        num_el_read = c_size_t()
        res |= cbflib.cbf_get_integerarray(
            handle, byref(binary_id), byref(data), el_size, 1, c_size_t(num_el), byref(num_el_read)
        )
        if res != 0:
            logger.error('MiniCBF Image data error: %s' % (_format_error(res),))

    data = numpy.fromstring(data, dtype=el_type).reshape(*header['detector_size'][::-1])

    res |= cbflib.cbf_free_goniometer(goniometer)
    res |= cbflib.cbf_free_detector(detector)
    res |= cbflib.cbf_free_handle(handle)

    return header, data


class CBFDataSet(DataSet):
    name = 'CBF Area Detector Image'
    def __init__(self, filename, header_only=False):
        super(CBFDataSet, self).__init__()
        self.filename = filename
        self.name = os.path.splitext(os.path.basename(self.filename))[0]
        self.current_frame = 1
        self.raw_header, self.raw_data = read_cbf(filename)
        self.read_dataset()

    def read_dataset(self):
        self.header = {}
        self.header.update(self.raw_header)
        self.header.update({
            'format': 'CBF',
            'name': self.name,
            'dataset': utils.file_sequences(self.filename),
        })

        if self.header['dataset']:
            self.current_frame = self.header['dataset']['current']
            self.header['name'] = self.header['dataset']['label']
            self.header['dataset'].update({
                'start_angle': (
                        self.header['start_angle'] - self.header['delta_angle'] * (
                            self.header['dataset']['current'] - 1)
                )
            })

        self.data = self.raw_data.view(numpy.int32)
        stats_subset = self.data[:self.data.shape[0] // 2, :self.data.shape[1] // 2]
        valid = (stats_subset > 0) & (stats_subset <= self.header['saturated_value'])
        self.stats_data = stats_subset[valid]

        self.header['average_intensity'], self.header['std_dev'] = numpy.ravel(cv2.meanStdDev(self.stats_data))
        self.header['min_intensity'] = self.stats_data.min()
        self.header['max_intensity'] = self.stats_data.max()
        self.header['overloads'] = 4 * (self.stats_data == self.header['saturated_value']).sum()
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
                self.raw_header, self.raw_data, = read_cbf(filename, True)
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

__all__ = ['CBFDataSet']
