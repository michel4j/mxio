import os
import re
from pathlib import Path
from typing import Tuple, Union, BinaryIO

import numpy
from numpy.typing import NDArray

from mxio import DataSet, XYPair

__all__ = [
    "SMVDataSet"
]

DATA_TYPES = {
    "unsigned_short": 'u2',
    "unsigned_int": 'u4',
    "signed_short": 'i2',
    "signed_int": 'i4',
}


class SMVDataSet(DataSet):

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        magic = b'{\nHEADER_BYTES='
        if file.read(len(magic)) == magic:
            return "SMV Area Detector Image",

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        header = {'format': 'SMV'}
        with open(filename, 'rb') as file:
            file.readline()
            header_line = file.readline().decode('utf-8')
            header_size = int(re.match(r"^HEADER_BYTES=\s+(?P<size>\d+);$", header_line).group('size'))
            file.seek(0)
            header_text = file.read(header_size).decode('utf-8')
            pattern = re.compile(r'\n(?P<key>\w+)=(?P<value>.+);')
            info = {
                match.groupdict()['key'].lower(): match.groupdict()['value'].strip()
                for match in pattern.finditer(header_text)
            }

            data_type_str = DATA_TYPES[info.get('type', "unsigned_short")]
            endian = '>' if info.get("byte_order") == 'big_endian' else '<'
            data_type = numpy.dtype(f'{data_type_str}')
            pixel_size = float(info['pixel_size'])
            x_center = float(info['beam_center_x']) / pixel_size
            y_center = float(info['beam_center_y']) / pixel_size

            header['delta_angle'] = float(info['osc_range'])
            header['distance'] = float(info['distance'])
            header['wavelength'] = float(info['wavelength'])
            header['exposure'] = float(info['time'])
            header['pixel_size'] = XYPair(pixel_size, pixel_size)
            header['size'] = XYPair(int(info['size1']), int(info['size2']))
            header['center'] = XYPair(x_center, header['size'].y - y_center)
            header['start_angle'] = float(info['osc_start'])
            header['two_theta'] = float(info.get('twotheta', 0.0))
            header['filename'] = os.path.basename(filename)
            header['cutoff_value'] = int(info.get('ccd_image_saturation', 2 ** (8 * data_type.itemsize) - 1))

            header['detector'] = {
                (2048, 0.050): 'ADSC Q105',
                (2048, 0.102): 'ADSC Q210',
                (4096, 0.051): 'ADSC Q210',
                (2304, 0.082): 'ADSC Q4',
                (1152, 0.163): 'ADSC Q4',
                (3072, 0.103): 'ADSC Q315',
                (6144, 0.051): 'ADSC Q315',
                (2048, 0.078): 'NOIR-1',
                (1500, 0.200): 'RAXIS4',
                (3000, 0.100): 'RAXIS4',
                (6000, 0.050): 'RAXIS4',
                (1024, 0.090): 'SATURN 92',
                (1042, 0.090): 'SATURN 92',
                (1024, 0.100): 'BRANDEIS CCD',
                (2080, 0.050): 'BRANDEIS CCD',
            }[(header['size'].x, round(pixel_size, 3))]

            num_elements = header['size'].x * header['size'].y
            data_size = num_elements * data_type.itemsize
            file.seek(0)
            file.read(header_size)
            raw_data = file.read(data_size)

        data = numpy.frombuffer(raw_data, dtype=data_type).reshape(header['size'].y, header['size'].x)
        if endian == '>':
            data = data.byteswap()
        return header, data
