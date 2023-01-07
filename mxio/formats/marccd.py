import os
import cv2
import math
import struct
from pathlib import Path
from typing import Tuple, Union, BinaryIO
from numpy.typing import NDArray

from mxio import DataSet, XYPair

__all__ = [
    "MarCCDDataSet"
]


class MarCCDDataSet(DataSet):

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        magic = b'MMX\0\0\0\0\0\0\0\0\0\0\0\0\0'
        file.seek(0x404)
        if file.read(len(magic)) == magic:
            return "MAR Area Detector Image",

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        # Read MarCCD header
        header_format = 'I16s39I80x'  # 256 bytes
        statistics_format = '3Q7I9I40x128H'  # 128 + 256 bytes
        goniostat_format = '28i16x'  # 128 bytes
        detector_format = '5i9i9i9i'  # 128 bytes
        source_format = '10i16x10i32x'  # 128 bytes

        with open(Path(filename), 'rb') as file:
            file.seek(1024)
            header_fields = struct.unpack(header_format, file.read(256))
            statistics_fields = struct.unpack(statistics_format, file.read(128 + 256))
            goniostat_fields = struct.unpack(goniostat_format, file.read(128))
            detector_fields = struct.unpack(detector_format, file.read(128))
            source_fields = struct.unpack(source_format, file.read(128))

        header = {
            'pixel_size': XYPair(detector_fields[1] / 1e6, detector_fields[1] / 1e6),
            'center': XYPair(goniostat_fields[1] / 1e3, goniostat_fields[2] / 1e3),
            'size': XYPair(header_fields[17], header_fields[18]),
            'distance': goniostat_fields[0] / 1e3,
            'two_theta': (goniostat_fields[7] / 1e3) * math.pi / -180.0,
            'exposure': goniostat_fields[4] / 1e3,
            'wavelength': source_fields[3] / 1e5,
            'start_angle': goniostat_fields[(7 + goniostat_fields[23])] / 1e3,
            'delta_angle': goniostat_fields[24] / 1e3,
            'minimum': statistics_fields[3],
            'maximum': statistics_fields[4],
            'average': statistics_fields[5] / 1e3,
            'overloads': statistics_fields[8],
            'cutoff_value': header_fields[23],
            'filename': os.path.basename(filename)
        }

        det_mm = int(round(header['pixel_size'].x * header['size'].x))
        header['detector'] = f'Rayonix MX{det_mm:d}'
        header['format'] = 'TIFF'

        data = cv2.imread(str(filename), -1)
        return header, data

