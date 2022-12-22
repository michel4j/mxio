import os
import cv2
import math
import struct
from pathlib import Path
from typing import Tuple, Union
from numpy.typing import ArrayLike

from mxio.dataset import DataSet, ImageFrame, XYPair

__all__ = [
    "MarCCDDataSet"
]


class MarCCDDataSet(DataSet):
    magic = (
        {'offset': 0x404, "magic": b'MMX\0\0\0\0\0\0\0\0\0\0\0\0\0', "name": "MAR Area Detector Image"},
    )

    def get_frame(self, index: int) -> Union[ImageFrame, None]:
        if any(index in range(sweep[0], sweep[1]+1) for sweep in self.sweeps):
            file_name = self.directory.joinpath(self.template.format(field=index))
            header, data = self._read_mar_file(file_name)
            frame = ImageFrame(**header, data=data)
            self.frame = frame
            self.index = index
            return self.frame

    def next_frame(self) -> Union[ImageFrame, None]:
        return self.get_frame(self.index + 1)

    def prev_frame(self) -> Union[ImageFrame, None]:
        return self.get_frame(self.index - 1)

    @staticmethod
    def _read_mar_file(filename: Union[str, Path]) -> Tuple[dict, ArrayLike]:
        header = {}

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

        header['pixel_size'] = XYPair(detector_fields[1] / 1e6, detector_fields[1] / 1e6)
        header['center'] = XYPair(goniostat_fields[1] / 1e3, goniostat_fields[2] / 1e3)
        header['size'] = XYPair(header_fields[17], header_fields[18])
        header['distance'] = goniostat_fields[0] / 1e3
        header['two_theta'] = (goniostat_fields[7] / 1e3) * math.pi / -180.0
        header['exposure'] = goniostat_fields[4] / 1e3
        header['wavelength'] = source_fields[3] / 1e5
        header['start_angle'] = goniostat_fields[(7 + goniostat_fields[23])] / 1e3
        header['delta_angle'] = goniostat_fields[24] / 1e3
        header['minimum'] = statistics_fields[3]
        header['maximum'] = statistics_fields[4]
        header['average'] = statistics_fields[5] / 1e3
        header['overloads'] = statistics_fields[8]
        header['saturated_value'] = header_fields[23]
        header['filename'] = os.path.basename(filename)

        det_mm = int(round(header['pixel_size'].x * header['size'].x))
        header['detector'] = f'MX{det_mm:d}'
        header['format'] = 'TIFF'

        data = cv2.imread(str(filename), -1)
        return header, data

