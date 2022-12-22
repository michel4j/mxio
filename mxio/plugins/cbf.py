import os
import re

import ctypes as ct
import numpy
from enum import Enum
from pathlib import Path
from typing import Tuple, Union
from numpy.typing import ArrayLike

from mxio.dataset import DataSet, ImageFrame, XYPair
from mxio import parser
__all__ = [
    "CBFDataSet"
]


# Define CBF constants
class ErrorType:
    FORMAT = 0x00000001  # 1
    ALLOC = 0x00000002  # 2
    ARGUMENT = 0x00000004  # 4
    ASCII = 0x00000008  # 8
    BINARY = 0x00000010  # 16
    BIT_COUNT = 0x00000020  # 32
    END_OF_DATA = 0x00000040  # 64
    FILE_CLOSE = 0x00000080  # 128
    FILE_OPEN = 0x00000100  # 256
    FILE_READ = 0x00000200  # 512
    FILE_SEEK = 0x00000400  # 1024
    FILE_TELL = 0x00000800  # 2048
    FILE_WRITE = 0x00001000  # 4096
    IDENTICAL = 0x00002000  # 8192
    NOT_FOUND = 0x00004000  # 16384
    OVERFLOW = 0x00008000  # 32768
    UNDEFINED = 0x00010000  # 65536
    NOT_IMPLEMENTED = 0x00020000  # 131072
    NO_COMPRESSION = 0x00040000  # 262144


class Flags:
    PLAIN_HEADERS = 0x0001  # Use plain ASCII headers
    MIME_HEADERS = 0x0002  # Use MIME headers
    MSG_NO_DIGEST = 0x0004  # Do not check message digests
    MSG_DIGEST = 0x0008  # Check message digests
    MSG_DIGEST_NOW = 0x0010  # Check message digests immediately
    MSG_DIGEST_WARN = 0x0020  # Warn on message digests immediately
    PAD_1K = 0x0020  # Pad binaries with 1023 0's
    PAD_2K = 0x0040  # Pad binaries with 2047 0's
    BYTE_OFFSET = 0x0070
    PAD_4K = 0x0080  # Pad binaries with 4095 0's

ERROR_MESSAGES = {
    ErrorType.FORMAT: 'Invalid File Format',
    ErrorType.ALLOC: 'Memory Allocation Error',
    ErrorType.ARGUMENT: 'Invalid function arguments',
    ErrorType.ASCII: 'Value is ASCII (not binary)',
    ErrorType.BINARY: 'Value is binary (not ASCII)',
    ErrorType.BIT_COUNT: 'Expected number of bits does not match actual number written',
    ErrorType.END_OF_DATA: 'End of data was reached before end of array',
    ErrorType.FILE_CLOSE: 'File close error',
    ErrorType.FILE_OPEN: 'File open error',
    ErrorType.FILE_READ: 'File read error',
    ErrorType.FILE_SEEK: 'File seek error',
    ErrorType.FILE_TELL: 'File tell error',
    ErrorType.FILE_WRITE: 'File write error',
    ErrorType.IDENTICAL: 'Data block with identical name already exists',
    ErrorType.NOT_FOUND: 'Data block/category/column/row does not exist',
    ErrorType.OVERFLOW: 'Value overflow error. The value has been truncated',
    ErrorType.UNDEFINED: 'Requested number is undefined',
    ErrorType.NOT_IMPLEMENTED: 'Requested functionality is not implemented',
    ErrorType.NO_COMPRESSION: 'No compression',
}


DATA_TYPES = {
    "unsigned 16-bit integer": (ct.c_uint16, 'F;16', 'F;16B'),
    "unsigned 32-bit integer": (ct.c_uint32, 'F;32', 'F;32B'),
    "signed 16-bit integer": (ct.c_int16, 'F;16S', 'F;16BS'),
    "signed 32-bit integer": (ct.c_int32, 'F;32S', 'F;32BS'),
}

ELEMENT_TYPES = {
    "signed 32-bit integer": ct.c_int32,
}


PILATUS_HEADER_SPECS = """
root:
  fields:
    - "# Detector: <str:detector>, S/N <slug:serial_number>"
    - "# Pixel_size <float:pixel_size> m x <float:pixel_size> m"
    - "# Exposure_period <float:exposure> s"
    - "# Count_cutoff <int:saturated_value> counts"
    - "# Wavelength <float:wavelength> A"
    - "# Detector_distance <float:distance> m"
    - "# Beam_xy (<float:center>, <float:center>) pixels"
    - "# Start_angle <float:start_angle> deg"
    - "# Angle_increment <float:delta_angle> deg."
    - "# Detector_2theta <float:two_theta> deg."
    - "# Silicon sensor, thickness <float:sensor_thickness> m"

"""

SLS_HEADER_SPECS = """
root:
  fields:
    - "# Detector: <str:detector>, SN <slug:serial_number>, <str:beamline>"
    - "# Pixel_size <float:pixel_size> m x <float:pixel_size> m"
    - "# Exposure_period <float:exposure> s"
    - "# Count_cutoff <int:saturated_value> counts"
    - "# Wavelength <float:wavelength> A"
    - "# Detector_distance <float:distance> m"
    - "# Beam_xy (<float:center>, <float:center>) pixels"
    - "# Start_angle <float:start_angle> deg"
    - "# Angle_increment <float:delta_angle> deg."
    - "# Detector_2theta <float:two_theta> deg."
    - "# Silicon sensor, thickness <float:sensor_thickness> m"
"""

HEADER_SPECS = {
    "PILATUS_1.2": PILATUS_HEADER_SPECS,
    "SLS_1.0": SLS_HEADER_SPECS,
}


def _format_error(code):
    errors = []
    for k, v in ERROR_MESSAGES.items():
        if (code | k) == code:
            errors.append(v)
    return ', '.join(errors)


def max_int(t):
    v = t(2 ** (8 * ct.sizeof(t)) - 1).value
    if v == -1:  # signed
        return ct.c_double(2 ** (8 * ct.sizeof(t) - 1) - 1)
    else:  # unsigned
        return ct.c_double(2 ** (8 * ct.sizeof(t) - 1))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / numpy.linalg.norm(vector)


cbflib = ct.cdll.LoadLibrary('libcbf.so.0')
libc = ct.cdll.LoadLibrary('libc.so.6')

# LIBC Arg and return types
libc.fopen.argtypes = [ct.c_char_p, ct.c_char_p]
libc.fopen.restype = ct.c_void_p

# CBF Arg and return types
cbflib.cbf_make_handle.argtypes = [ct.c_void_p]
cbflib.cbf_free_handle.argtypes = [ct.c_void_p]
cbflib.cbf_read_file.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
cbflib.cbf_read_widefile.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
cbflib.cbf_get_wavelength.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
cbflib.cbf_get_integration_time.argtypes = [ct.c_void_p, ct.c_uint, ct.POINTER(ct.c_double)]
cbflib.cbf_get_rotation_range.argtypes = [ct.c_void_p, ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
cbflib.cbf_construct_goniometer.argtypes = [ct.c_void_p, ct.c_void_p]
cbflib.cbf_write_file.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
cbflib.cbf_construct_detector.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
cbflib.cbf_construct_reference_detector.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
cbflib.cbf_require_reference_detector.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
cbflib.cbf_read_template.argtypes = [ct.c_void_p, ct.c_void_p]
cbflib.cbf_get_pixel_size.argtypes = [ct.c_void_p, ct.c_uint, ct.c_int, ct.POINTER(ct.c_double)]
cbflib.cbf_get_detector_distance.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
cbflib.cbf_get_detector_id.argtypes = [ct.c_void_p, ct.c_uint, ct.c_void_p]
cbflib.cbf_select_datablock.argtypes = [ct.c_void_p, ct.c_uint]
cbflib.cbf_count_datablocks.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint)]
cbflib.cbf_find_datablock.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_find_category.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_find_column.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_datablock_name.argtypes = [ct.c_void_p, ct.c_void_p]
cbflib.cbf_get_overload.argtypes = [ct.c_void_p, ct.c_uint, ct.POINTER(ct.c_double)]
cbflib.cbf_new_datablock.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_new_category.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_new_column.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_set_value.argtypes = [ct.c_void_p, ct.c_char_p]
cbflib.cbf_get_beam_center.argtypes = [
    ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
]
cbflib.cbf_get_image_size.argtypes = [
    ct.c_void_p, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)
]
cbflib.cbf_get_detector_normal.argtypes = [
    ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
]
cbflib.cbf_get_rotation_axis.argtypes = [
    ct.c_void_p, ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
]
cbflib.cbf_get_image.argtypes = [
    ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_size_t, ct.c_size_t
]
cbflib.cbf_parse_mimeheader.argtypes = [
    ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_long),
    ct.c_void_p, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
    ct.c_void_p, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t),
    ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)
]
cbflib.cbf_get_integerarray.argtypes = [
    ct.c_void_p, ct.POINTER(ct.c_int), ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_size_t, ct.POINTER(ct.c_size_t)
]
cbflib.cbf_set_integerarray_wdims.argtypes = [
    ct.c_void_p, ct.c_uint, ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_size_t, ct.c_char_p, ct.c_size_t,
    ct.c_size_t, ct.c_size_t, ct.c_size_t
]


class CBFDataSet(DataSet):
    magic = (
        {'offset': 0, "magic": b'###CBF: VERSION', "name": "CBF Area Detector Image"},
    )

    def get_frame(self, index: int) -> Union[ImageFrame, None]:
        if any(index in range(sweep[0], sweep[1] + 1) for sweep in self.sweeps):
            file_name = self.directory.joinpath(self.template.format(field=index))
            header, data = self._read_cbf_file(file_name)
            frame = ImageFrame(**header, data=data)
            self.frame = frame
            self.index = index
            return self.frame

    def next_frame(self) -> Union[ImageFrame, None]:
        return self.get_frame(self.index + 1)

    def prev_frame(self) -> Union[ImageFrame, None]:
        return self.get_frame(self.index - 1)

    @classmethod
    def save_frame(cls, file_path: Union[os.PathLike, str], frame: ImageFrame, serial_number: str = '###-###'):
        header_text = (
            f"\n"
            f"# Detector: {frame.detector}, S/N {serial_number}\n"
            f"# Pixel_size {frame.pixel_size.x * 1000:0.0f}e-6 m x {frame.pixel_size.y*1000:0.0f}e-6 m\n"
            f"# Silicon sensor, thickness {frame.sensor_thickness/1000:0.6f} m\n"
            f"# Exposure_time {frame.exposure:0.7f} s\n"
            f"# Exposure_period {frame.exposure:0.7f} s\n"
            f"# Count_cutoff {frame.saturated_value:0.0f} counts\n"
            f"# Wavelength {frame.wavelength:0.5f} A\n"
            f"# Flux 0.000000\n"
            f"# Filter_transmission 1.0000\n"
            f"# Detector_distance {frame.distance:0.5f} m\n"
            f"# Beam_xy ({frame.center.x}, {frame.center.y}) pixels\n"
            f"# Start_angle {frame.start_angle:0.4f} deg.\n"
            f"# Angle_increment {frame.delta_angle:0.4f} deg.\n"
            f"# Detector_2theta {frame.two_theta:0.4f} deg.\n"
        ).encode('utf-8')
        header_content = header_text + (4096 - len(header_text)) * b'\0'  # pad header to 4096 bytes

        # create CBF handle
        cbf = ct.c_void_p()
        cbflib.cbf_make_handle(ct.byref(cbf))
        cbflib.cbf_new_datablock(cbf, b"image_1")
        fh = libc.fopen(str(file_path).encode('utf-8'), b"wb")

        # Write miniCBF header
        cbflib.cbf_new_category(cbf, b"array_data")
        cbflib.cbf_new_column(cbf, b"header_convention")
        cbflib.cbf_set_value(cbf, b"PILATUS_1.2")
        cbflib.cbf_new_column(cbf, b"header_contents")
        cbflib.cbf_set_value(cbf, header_content)

        # Write the image data
        cbflib.cbf_new_category(cbf, b"array_data")
        cbflib.cbf_new_column(cbf, b"data")

        data = ct.create_string_buffer(frame.data.tobytes())
        cbflib.cbf_set_integerarray_wdims(
            cbf,
            Flags.BYTE_OFFSET,
            1,  # binary id
            ct.byref(data),
            ct.c_size_t(frame.data.itemsize),
            1,  # signed
            frame.size.x * frame.size.y,
            b"little_endian",
            ct.c_size_t(frame.size.x),
            ct.c_size_t(frame.size.y),
            0,
            0
        )
        cbflib.cbf_write_file(cbf, fh, 1, 0, Flags.MSG_DIGEST | Flags.MIME_HEADERS | Flags.PAD_4K, 0)
        cbflib.cbf_free_handle(cbf)

    @staticmethod
    def _read_cbf_file(filename: Union[str, Path]) -> Tuple[dict, ArrayLike]:
        handle = ct.c_void_p()
        goniometer = ct.c_void_p()
        detector = ct.c_void_p()

        # make the handle and read the file
        result = cbflib.cbf_make_handle(ct.byref(handle))
        file_pointer = libc.fopen(str(filename).encode('utf-8'), b"rb")
        result |= cbflib.cbf_read_template(handle, file_pointer)
        result |= cbflib.cbf_construct_goniometer(handle, ct.byref(goniometer))
        result |= cbflib.cbf_require_reference_detector(handle, ct.byref(detector), 0)

        # read mime
        pattern = re.compile(r'^(.+):\s+(.+)$')
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

        with open(filename, 'rb') as file:
            # find start of binary header
            i = 0
            bin_found = False
            while not bin_found and i < 512:
                line = file.readline().decode()
                bin_found = bin_st.match(line)
                i += 1

            # extract binary header
            line = file.readline().decode()
            while line.strip() != '':
                m = pattern.match(line)
                if m:
                    mime_header[m.group(1)] = parse_tokens[m.group(1)](m.group(2).replace('"', '').strip())
                line = file.readline().decode()

        # read header
        header = {
            'format': 'CBF',
            'two_theta': 0.0,
            'filename': os.path.basename(filename)
        }

        wavelength = ct.c_double(1.0)
        result = cbflib.cbf_get_wavelength(handle, ct.byref(wavelength))
        header['wavelength'] = wavelength.value

        x_size = ct.c_size_t(mime_header.get('X-Binary-Size-Fastest-Dimension', 0))
        y_size = ct.c_size_t(mime_header.get('X-Binary-Size-Second-Dimension', 0))
        header['size'] = XYPair(x_size.value, y_size.value)

        pixel_size = ct.c_double(1.0)
        result |= cbflib.cbf_get_pixel_size(handle, 0, 1, ct.byref(pixel_size))
        header['pixel_size'] = XYPair(pixel_size.value, pixel_size.value)

        distance = ct.c_double(999.0)
        result |= cbflib.cbf_get_detector_distance(detector, ct.byref(distance))
        header['distance'] = distance.value

        x_center, y_center = ct.c_double(0.0), ct.c_double(0.0)
        ix, iy = ct.c_double(x_size.value / 2.0), ct.c_double(y_size.value / 2.0)
        result |= cbflib.cbf_get_beam_center(detector, ct.byref(ix), ct.byref(iy), ct.byref(x_center), ct.byref(y_center))
        header['center'] = XYPair(ix.value, iy.value)

        exposure = ct.c_double(0.0)
        result |= cbflib.cbf_get_integration_time(handle, 0, exposure)
        header['exposure'] = exposure.value

        start_angle, delta_angle = ct.c_double(0.0), ct.c_double(0.0)
        result |= cbflib.cbf_get_rotation_range(goniometer, 0, ct.byref(start_angle), ct.byref(delta_angle))
        header['start_angle'] = start_angle.value
        header['delta_angle'] = delta_angle.value

        element_type = DATA_TYPES[mime_header.get('X-Binary-Element-Type', 'signed 32-bit integer')][0]
        ovl = max_int(element_type)
        result |= cbflib.cbf_get_overload(handle, 0, ct.byref(ovl))
        header['saturated_value'] = ovl.value

        # FIXME Calculate actual two_theta from the beam direction and detector normal
        # vec_x, vec_y, vec_z = ct.c_double(0.0), ct.c_double(0.0), ct.c_double(0.0)
        # result |= cbflib.cbf_get_detector_normal(detector, ct.byref(vec_x), ct.byref(vec_y), ct.byref(vec_x))
        # detector_norm = numpy.array([vec_x.value, vec_y.value, vec_z.value])
        #
        # result |= cbflib.cbf_get_rotation_axis(goniometer, 0, ct.byref(vec_x), ct.byref(vec_y), ct.byref(vec_x))
        # rot_axis = numpy.array([vec_x.value, vec_y.value, vec_z.value])
        # header['two_theta'] = 0.0

        detector_name = ct.c_char_p()
        result |= cbflib.cbf_get_detector_id(handle, 0, ct.byref(detector_name))
        header['detector'] = detector_name.value if detector_name.value else "Unknown"

        # handle XDS Special cbf files
        if header['distance'] == 999.0 and header['delta_angle'] == 0.0 and header['exposure'] == 0.0:
            mini_cbf_type = ct.c_char_p()
            mini_cbf_header = ct.c_char_p()

            result = cbflib.cbf_select_datablock(handle, ct.c_uint(0))
            result |= cbflib.cbf_find_category(handle, b"array_data")
            result |= cbflib.cbf_find_column(handle, b"header_convention")
            result |= cbflib.cbf_get_value(handle, ct.byref(mini_cbf_type))
            result |= cbflib.cbf_find_column(handle, b"header_contents")
            result |= cbflib.cbf_get_value(handle, ct.byref(mini_cbf_header))

            if result == 0 and mini_cbf_type.value != b'XDS special':
                spec_text = HEADER_SPECS[mini_cbf_type.value.decode()]
                info = parser.parse_text((mini_cbf_header.value).decode(), spec_text=spec_text)

                pixel_size = list(map(lambda v: round(v*1000,5), info['pixel_size']))
                header['detector'] = info['detector'].strip().replace(' ', '')
                header['two_theta'] = 0 if not info['two_theta'] else round(info['two_theta'], 2)
                header['pixel_size'] = XYPair(*pixel_size)
                header['exposure'] = info['exposure']
                header['wavelength'] = info['wavelength']
                header['distance'] = info['distance'] * 1000
                header['center'] = XYPair(*info['center'])
                header['start_angle'] = info['start_angle']
                header['delta_angle'] = info['delta_angle']
                header['saturated_value'] = info['saturated_value']
                header['sensor_thickness'] = info['sensor_thickness'] * 1000

        num_elements = header['size'].x * header['size'].y
        mime_params = DATA_TYPES[mime_header.get('X-Binary-Element-Type', 'signed 32-bit integer')]
        element_type = mime_params[0]
        element_size = ct.sizeof(element_type)
        data = ct.create_string_buffer(num_elements * element_size)
        result = cbflib.cbf_get_image(handle, 0, 0, ct.byref(data), element_size, 1, header['size'].x, header['size'].y)
        if result != 0:
            # MiniCBF
            result = cbflib.cbf_select_datablock(handle, ct.c_uint(0))
            result |= cbflib.cbf_find_category(handle, b"array_data")
            result |= cbflib.cbf_find_column(handle, b"data")
            binary_id = ct.c_int(mime_header.get('X-Binary-ID', 1))
            elements_read = ct.c_size_t()
            result |= cbflib.cbf_get_integerarray(
                handle, ct.byref(binary_id), ct.byref(data), element_size, 1, ct.c_size_t(num_elements),
                ct.byref(elements_read)
            )

        data = numpy.frombuffer(data, dtype=element_type).reshape(header['size'].y, header['size'].x)

        result |= cbflib.cbf_free_goniometer(goniometer)
        result |= cbflib.cbf_free_detector(detector)
        result |= cbflib.cbf_free_handle(handle)

        # calculate extra statistics
        mask = data >= 0
        header['maximum'] = data[mask].max()
        header['average'] = data[mask].mean()
        header['minimum'] = data[mask].min()
        header['overloads'] = (data[mask] >= header['saturated_value']).sum()

        return header, data
