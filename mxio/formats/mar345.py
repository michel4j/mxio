import os
import numpy
import struct

from pathlib import Path
from typing import Tuple, Union, BinaryIO
from numpy.typing import NDArray

from mxio import DataSet, XYPair, parser

__all__ = [
    "MAR345DataSet"
]

HEADER_SPECS = {
    "magic": '1I',
    "width": '1i',
    "overloads": '1i',
    "compression": '1i',
    "mode": '1i',
    "size": '1i',
    "pixel_size": '2i',
    "wavelength": '1i',
    "distance": '1i',
    "start_phi": '1i',
    "end_phi": '1i',
    "start_omega": '1i',
    "end_omega": '1i',
    "chi": '1i',
    'two_theta': '1i',
    'description': '64s'
}

HEADER_LEXICON = {
    'fields': [
        "PROGRAM        <str:software>",
        "DATE           <str:date>",
        "SCANNER        <int:scanner>",
        "FORMAT         <str:format>",
        "HIGH           <int:overloads>",
        "PIXEL          LENGTH <int:pixel_size> HEIGHT <int:pixel_size>",
        "GAIN           <float:gain>",
        "WAVELENGTH     <float:wavelength>",
        "DISTANCE       <float:distance>",
        "RESOLUTION     <float:resolution>",
        "PHI            START <float:start_phi>  END <float:end_phi>  OSC <int:phi_scan>",
        "OMEGA          START <float:start_omega>  END <float:end_omega>  OSC <int:omega_scan>",
        "CHI            <float:chi>",
        "TWOTHETA       <float:two_theta>",
        "CENTER         X <float:center>  Y <float:center>",
        "MODE           <str:exposure_mode>",
        "TIME           <float:exposure>",
        "INTENSITY      MIN <int:minimum>  MAX <int:maximum>  AVE <float:average>  SIG <float:sigma>",
        "COLLIMATOR     WIDTH <float:beam_size>  HEIGHT <float:beam_size>",
        "DETECTOR       <str:detector>",
    ]
}


class MAR345DataSet(DataSet):

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:

        try:
            from pycbf.img import Img
        except ImportError:
            pass
        else:
            magic = file.read(2)
            if magic == b'\x04\xd2':
                return "MAR345 Area Detector Image", "Big-Endian"
            elif magic == b'\xd2\x04':
                return "MAR345 Area Detector Image", "Little-Endian"

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
        try:
            from pycbf.img import Img
        except ImportError:
            return {}, None

        # Read MAR345 header
        with open(Path(filename), 'rb') as file:
            endian = '>' if "Big-Endian" in self.tags else '<'
            info = {
                key: struct.unpack(f"{endian}{format_str}", file.read(struct.calcsize(f"{endian}{format_str}")))
                for key, format_str in HEADER_SPECS.items()
            }

            header_lines = []
            line = ''
            while line.strip() != 'END OF HEADER':
                try:
                    line = file.readline()
                    line = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    line = ''
                if line:
                    header_lines.append(line)
            header_text = "\n".join(header_lines)

            keywords = parser.parse_text(HEADER_LEXICON, header_text)
            if keywords.get('phi_scan'):
                start_angle = keywords['start_phi']
                delta_angle = keywords['end_phi'] - start_angle
            else:
                start_angle = keywords['start_omega']
                delta_angle = keywords['end_omega'] - start_angle

            header = {
                'format': 'MAR345',
                'filename': os.path.basename(filename),
                'pixel_size': XYPair(info['pixel_size'][0]/1000, info['pixel_size'][1]/1000),
                'center': XYPair(*keywords['center']),
                'size': XYPair(info['width'][0], info['width'][0]),
                'distance': info['distance'][0]/1000,
                'two_theta': info['two_theta'][0]/1000,
                'exposure': keywords['exposure'],
                'wavelength': keywords['wavelength'],
                'start_angle': start_angle,
                'delta_angle': delta_angle,
                'detector': keywords.get('detector', 'MAR345').upper(),
                'cutoff_value': 65535*2,
                'overloads': info['overloads'][0],
                'average': keywords['average'],
                'maximum': keywords['maximum'],
                'sigma': keywords['sigma'],
                'minimum': keywords['minimum'],
            }

            file.seek(4096)

            # read image data
            img = Img()
            swap = 1 if endian == '>' else 0
            img.read_mar345data(file, (header['size'].x, header['size'].y, info['overloads'][0], swap))
            data = img.image

        return header, data
