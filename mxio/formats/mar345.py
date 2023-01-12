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
        magic = file.read(2)
        if magic == b'\x04\xd2':
            return "MAR345 Area Detector Image", "Big-Endian"
        elif magic == b'\xd2\x04':
            return "MAR345 Area Detector Image", "Little-Endian"

    def read_file(self, filename: Union[str, Path]) -> Tuple[dict, NDArray]:
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
                    print(line)
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
                'cutoff_value': 65535,
                'overloads': info['overloads'][0],
                'average': keywords['average'],
                'maximum': keywords['maximum'],
                'sigma': keywords['sigma'],
                'minimum': keywords['minimum'],
            }

            high_bytes = int(info['overloads'][0]/8 + 0.875) * 64
            file.seek(4096 + high_bytes)
            raw_data = file.read()

            data_type = numpy.dtype(f'{endian}u2')

        data = unpack_bits(raw_data, header['size'].x, header['size'].y)

        return header, data.view(data_type)


def unpack_bits(raw_data: bytes, x: int, y: int) -> NDArray:
    """
    Unpack a series of bytes using the RLE algorithm
    :param raw_data: #byte string to be decompressed
    :param x: x-size
    :param y: y-size
    :return: decompressed array
    """
    setbits = numpy.array([
        0x00000000, 0x00000001, 0x00000003, 0x00000007,
		0x0000000F, 0x0000001F, 0x0000003F, 0x0000007F,
		0x000000FF, 0x000001FF, 0x000003FF, 0x000007FF,
		0x00000FFF, 0x00001FFF, 0x00003FFF, 0x00007FFF,
		0x0000FFFF, 0x0001FFFF, 0x0003FFFF, 0x0007FFFF,
		0x000FFFFF, 0x001FFFFF, 0x003FFFFF, 0x007FFFFF,
		0x00FFFFFF, 0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF,
		0x0FFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF,
        0xFFFFFFFF
    ], dtype=numpy.int32)

    def shift_left(v, n): return (v & setbits[32 - n]) << n
    def shift_right(v, n): return (v >> n) & setbits[32 - n]

    pos = 0
    pixel = 0
    window = 0
    spill = 0
    bitdecode = [0, 4, 5, 6, 7, 8, 16, 32]

    valids = numpy.int32(0)
    spillbits = numpy.int32(0)
    total = x * y

    byte_array = numpy.frombuffer(raw_data, dtype=numpy.ubyte)
    data = numpy.zeros((total, ), dtype=numpy.uint16)

    try:
        while pixel < total:
            if valids < 6:
                if spillbits > 0:
                    window |= shift_left(spill, valids)
                    valids += spillbits
                    spillbits = 0
                else:
                    spill = numpy.int32(byte_array[pos]); pos += 1
                    spillbits = 8
            else:
                pixnum = 1 << (window & setbits[3])
                window = shift_right(window, 3)
                bitnum = bitdecode[window & setbits[3]]
                window = shift_right(window, 3)
                valids -= 6
                while pixnum > 0 and pixel < total:
                    if valids < bitnum:
                        if spillbits > 0:
                            window |= shift_left(spill, valids)
                            if 32 - valids > spillbits:
                                valids += spillbits
                                spillbits = 0
                            else:
                                usedbits = 32 - valids
                                spill = shift_right(spill, usedbits)
                                spillbits -= usedbits
                                valids = 32
                        else:
                            spill = numpy.int32(byte_array[pos]); pos += 1
                            spillbits = 8
                    else:
                        pixnum -= 1
                        if bitnum == 0:
                            nextint = 0
                        else:
                            nextint = window & setbits[bitnum]
                            valids -= bitnum
                            window = shift_right(window, bitnum)
                            if (nextint & (1 << (bitnum - 1))) != 0:
                                nextint |= ~setbits[bitnum]
                        if pixel > x:
                            data[pixel] = (
                                nextint +
                                (data[pixel - 1] + data[pixel - x + 1] +  data[pixel - x] + data[pixel -x-1] + 2)/4
                            )
                            pixel += 1
                        elif pixel != 0:
                            data[pixel] = nextint + data[pixel - 1]
                            pixel += 1
                        else:
                            data[pixel] =  nextint
                            pixel += 1
    except Exception as e:
        print(pos, e)


    return data.reshape((y, x))

