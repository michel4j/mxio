import os
import numpy
import struct
from pathlib import Path
from typing import Tuple, Union, BinaryIO
from numpy.typing import NDArray

from mxio import DataSet, XYPair

__all__ = [
    "RAXISDataSet"
]

HEADER_SPECS = {
    "instrument": '10s',
    "version": '10s',
    "sample": '20s',
    "lattice": '12s6f12s',
    "mosaicity": '1f',
    "comments": '80s',
    "reserved_1": '84s',
    "date": '12s',
    "user": '20s',
    "src_target": '4s',
    "wavelength": '1f',
    "monochromator": '20s1f',
    "collimator": '20s',
    "filter": '4s',
    "distance": 'f',
    "src_voltage": '1f',
    "src_current": '1f',
    "focus": '12s',
    "optics": '80s',
    "ip_shape": '1l',
    "weissenberg": '1f',
    "reserved_2": '56s',
    "mount_axes": '4s4s',
    "angle": '1f',
    "start_angle": '1f',
    "end_angle": '1f',
    "index": '1l',
    "exposure": '1f',
    "center": '2f',
    "omega": '1f',
    "chi": '1f',
    "two_theta": '1f',
    "mu": '1f',
    "scan_template": '204s',
    "size": '2l',
    'pixel_size': '2f',
    "record_length": "1l",
    "record_count": "1l",
    "read_start": "1l",
    "ip_number": "1l",
    "photomultiplier_ratio": '1f',
    "fade_times": '2f',
    "host_type": '10s',
    "ip_type": '10s',
    "scan_codes": '3l',
    "pixel_shift": "1f",
    "intensity_ratio": '1f',
    "magic": '1l',
    "num_axes": '1l',
    "gonio_axes": '15f',
    "gonio_starts": '5f',
    "gonio_ends": '5f',
    "gonio_offsets": '5f',
    "scan_axis": '1l',
    "axis_names": '40s',
    "common": '16s20s20s9l20l20l1l768s'
}


class RAXISDataSet(DataSet):

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        magic = b'RAXIS'
        if file.read(len(magic)) == magic:
            return "RAXIS Area Detector Image",

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        # Read RAXIS header
        with open(Path(filename), 'rb') as file:
            # detect endianness
            file.seek(796)
            endian_test = struct.unpack('>l', file.read(struct.calcsize('>l')))[0]
            endian = '>' if endian_test < 20 else '<'
            file.seek(0)

            info = {
                key: struct.unpack(f"{endian}{format_str}", file.read(struct.calcsize(f"{endian}{format_str}")))
                for key, format_str in HEADER_SPECS.items()
            }
            data_type = numpy.dtype('u2')

            header = {
                'format': 'RAXIS',
                'filename': os.path.basename(filename),
                'pixel_size': XYPair(*info['pixel_size']),
                'center': XYPair(*info['center']),
                'size': XYPair(*info['size']),
                'distance': info['distance'][0],
                'two_theta': info['two_theta'][0],
                'exposure': info['exposure'][0],
                'wavelength': info['wavelength'][0],
                'start_angle': info['start_angle'][0],
                'delta_angle': info['end_angle'][0] - info['start_angle'][0],
                'detector': info['instrument'][0].decode('utf-8').strip(),
                'cutoff_value': int(2 ** (8 * data_type.itemsize) - 1)
            }

            num_elements = header['size'].x * header['size'].y
            data_size = num_elements * data_type.itemsize
            file.seek(-data_size, 2)
            raw_data = file.read(data_size)

        data = numpy.frombuffer(raw_data, dtype=data_type).reshape(header['size'].y, header['size'].x)
        if endian == '>':
            data = data.byteswap()

        return header, data

