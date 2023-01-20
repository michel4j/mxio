import hashlib
import json
from pathlib import Path
from typing import Tuple, Union, Sequence, BinaryIO

import lz4.block
import lz4.frame
import numpy
import zmq
from numpy.typing import NDArray

from mxio import DataSet, XYPair, HeaderAttrs, ImageFrame
from mxio.misc import bshuf


TYPES = {
    'uint16': numpy.dtype(numpy.int16),
    'uint32': numpy.dtype(numpy.int32),
    'uint64': numpy.dtype(numpy.int64),
    'int32': numpy.dtype(numpy.int32),
    'int64': numpy.dtype(numpy.int32),
}

HEADER_FIELDS = {
    'detector': 'description',
    'serial_number': 'detector_number',
    'two_theta': 'two_theta_start',
    'pixel_size': ('x_pixel_size', 'y_pixel_size'),
    'exposure': 'count_time',
    'wavelength': 'wavelength',
    'distance': 'detector_distance',
    'center': ('beam_center_x', 'beam_center_y'),
    'sensor_thickness': 'sensor_thickness',
    'size': ('x_pixels_in_detector', 'y_pixels_in_detector'),
    'count_cutoff': ('bit_depth_readout', 'nframes_sum'),
}

CONVERTERS = {
    'pixel_size': lambda v: float(v) * 1000,
    'exposure': float,
    'wavelength': float,
    'distance': lambda v: float(v) * 1000,
    'center': float,
    'num_frames': int,
    'sensor_thickness': lambda v: float(v) * 1000,
    'size': int,
}


class EigerStream(DataSet):
    global_header: dict

    @classmethod
    def identify(cls, file: BinaryIO, extension: str) -> Tuple[str, ...]:
        # does not support files so return empty tuple.
        return ()

    def setup(self):
        # dummy method, does nothing for Eiger Stream
        pass

    def read_file(self, filename: Union[str, Path]) -> Tuple[HeaderAttrs, NDArray]:
        # dummy method, does nothing for Eiger Stream
        raise RuntimeError("Eiger Stream does not support file reads.")

    def parse_header(self, message: Sequence[bytes]):
        """
        Read header information from Eiger Stream and update dataset header

        :param message:  multipart message of htype 'dheader-1.0'
        """
        preamble = json.loads(message[0])
        info = json.loads(message[1])

        metadata = {}
        for key, field in HEADER_FIELDS.items():
            converter = CONVERTERS.get(key, lambda v: v)
            try:
                if not isinstance(field, (tuple, list)):
                    metadata[key] = converter(info[field])
                else:
                    metadata[key] = tuple(converter(info[sub_field]) for sub_field in field)
            except ValueError:
                pass

        # try to find oscillation axis and parameters as first non-zero average
        for axis in ['chi', 'kappa', 'omega', 'phi']:
            if info.get('{}_increment'.format(axis), 0) > 0:
                metadata['start_angle'] = info['{}_start'.format(axis)]
                metadata['delta_angle'] = info['{}_increment'.format(axis)]
                break

        metadata['count_cutoff'] = 2**metadata['count_cutoff'][0] * metadata['count_cutoff'][1]

        self.name = f'series-{preamble["series"]}'
        self.size = info['nimages'] * info['ntrigger']
        self.series = numpy.arange(self.size) + 1
        self.global_header = metadata
        self.identifier = hashlib.blake2s(
            bytes(self.directory) + self.name.encode('utf-8'), digest_size=16
        ).hexdigest()

    def parse_image(self, message: Sequence[bytes]):
        """
        Parse image information from Eiger Stream state

        :param message:  multipart message from stream of htype='dimage-1.0'

        :returns: tuple of metadata, data
            metadata - information about the frame
            data - 2D array representing the image data
        """

        info = json.loads(message[0])
        metadata = json.loads(message[1])
        img_data = message[2]

        data_type = numpy.dtype(metadata['type'])
        shape = metadata['shape'][::-1]
        size = numpy.prod(shape)
        data_type = data_type.newbyteorder(metadata['encoding'][-1]) if metadata['encoding'][-1] in ['<', '>'] else data_type
        index = int(info['frame']) + 1
        header = {
            'format': 'Eiger Stream',
            'detector': self.global_header['detector'],
            'two_theta': self.global_header['two_theta'],
            'pixel_size': XYPair(*self.global_header['pixel_size']),
            'size': XYPair(*self.global_header['size']),
            'exposure': self.global_header['exposure'],
            'cutoff_value': 2.8e6, #self.global_header['count_cutoff'],
            'filename': self.name,
            'wavelength': self.global_header['wavelength'],
            'distance': self.global_header['distance'],
            'center': XYPair(*self.global_header['center']),
            'start_angle': self.global_header['start_angle'] + index * self.global_header['delta_angle'],
            'delta_angle': 0.0,
            'sensor_thickness': self.global_header['sensor_thickness'],
        }


        try:
            if metadata['encoding'].startswith('lz4'):
                arr_bytes = lz4.block.decompress(img_data, uncompressed_size=size * data_type.itemsize)
                raw_data = numpy.frombuffer(arr_bytes, dtype=data_type).reshape(*shape)
            elif metadata['encoding'].startswith('bs'):
                raw_data = bshuf.decompress_lz4(img_data[12:], shape, data_type)
            else:
                raise RuntimeError(f'Unknown encoding {metadata["encoding"]}')
            data = raw_data.view(TYPES[metadata['type']])
        except Exception as e:
            print(f'Error parsing frame data {e}')
            data = None

        self.set_frame(header, data, index)


def multiplexer(address, port):
    """
    Proxies the PUSH/PULL Stream from address and publishes it as a PUB/SUB Stream through port.
    This allows multiple ZMQ clients to access the same data.
    """

    context = zmq.Context()
    backend = context.socket(zmq.PULL)
    frontend = context.socket(zmq.PUB)

    try:
        backend.connect(address)
        frontend.bind(f"tcp://*:{port}")

        print(f'Starting ZMQ PULL Multiplexer proxying messages from {address} to {port} ...')
        zmq.device(zmq.STREAMER, frontend, backend)
    finally:
        frontend.close()
        backend.close()
        context.term()
