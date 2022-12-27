import json
import zmq

import cv2
import lz4.block
import lz4.frame
import numpy

from . import DataSet
from ..log import get_module_logger
from ..misc import bshuf

# Configure Logging
logger = get_module_logger('mxio')

SIZES = {
    'uint16': 2,
    'uint32': 4
}

TYPES = {
    'uint16': numpy.int16,
    'uint32': numpy.int32
}

HEADER_FIELDS = {
    'detector_type': 'description',
    'serial_number': 'detector_number',
    'two_theta': 'two_theta_start',
    'pixel_size': 'x_pixel_size',
    'exposure_time': 'count_time',
    'exposure_period': 'frame_time',
    'wavelength': 'wavelength',
    'distance': 'detector_distance',
    'beam_center': ('beam_center_x', 'beam_center_y'),
    'energy': 'photon_energy',
    'sensor_thickness': 'sensor_thickness',
    'detector_size': ('x_pixels_in_detector', 'y_pixels_in_detector'),

}
CONVERTERS = {
    'pixel_size': lambda v: float(v) * 1000,
    'exposure_time': float,
    'wavelength': float,
    'distance': lambda v: float(v) * 1000,
    'beam_center': float,
    'saturated_value': int,
    'num_frames': int,
    'date': 'data_collection_date',
    'energy': float,
    'sensor_thickness': lambda v: float(v) * 1000,
    'detector_size': int,
}


class EigerStream(DataSet):
    def __init__(self, name="Stream"):
        super().__init__()
        self.name = name
        self._start_angle = 0.0

    def read_dataset(self):
        # dummy method, does nothing for Eiger Stream
        pass

    def read_header(self, message):
        """
        Read header information from Eiger Stream and update dataset header

        :param message:  multipart message of htype 'dheader-1.0'
        """
        header = json.loads(message[1])
        metadata = {}
        for key, field in HEADER_FIELDS.items():
            converter = CONVERTERS.get(key, lambda v: v)
            try:
                if not isinstance(field, (tuple, list)):
                    metadata[key] = converter(header[field])
                else:
                    metadata[key] = tuple(converter(header[sub_field]) for sub_field in field)
            except ValueError:
                pass

        # try to find oscillation axis and parameters as first non-zero average
        for axis in ['chi', 'kappa', 'omega', 'phi']:
            if header.get('{}_increment'.format(axis), 0) > 0:
                metadata['rotation_axis'] = axis
                metadata['start_angle'] = header['{}_start'.format(axis)]
                metadata['delta_angle'] = header['{}_increment'.format(axis)]
                metadata['num_images'] = header['nimages']*header['ntrigger']
                metadata['total_angle'] = metadata['num_images'] * metadata['delta_angle']
                break
        self._start_angle = metadata['start_angle']
        self.header = metadata

    def read_image(self, message):
        """
        Read image information from Eiger Stream and update dataset data

        :param message:  multipart message from stream of htype='dimage-1.0'
        """
        metadata, data = self.parse_image(message)
        self.header.update(metadata)
        stats_data = data[(data >= 0) & (data < metadata['saturated_value'])]

        try:
            avg, stdev = numpy.ravel(cv2.meanStdDev(stats_data))
        except Exception as e:
            avg = stdev = 0.0
            logger.warning(f'Error calculating frame statistics: {e}')

        metadata.update({
            'average_intensity': avg,
            'std_dev': stdev,
            'min_intensity': stats_data.min(),
            'max_intensity': stats_data.max(),
        })
        self.data = data
        self.stats_data = stats_data
        self.header.update(metadata)

    def parse_image(self, message):
        """
        Parse image information from Eiger Stream without updating internal state

        :param message:  multipart message from stream of htype='dimage-1.0'

        :returns: tuple of metadata, data
            metadata - information about the frame
            data - 2D array representing the image data
        """

        info = json.loads(message[0])
        frame = json.loads(message[1])
        img_data = message[2]

        dtype = numpy.dtype(frame['type'])
        shape = frame['shape'][::-1]
        size = numpy.prod(shape)
        dtype = dtype.newbyteorder(frame['encoding'][-1]) if frame['encoding'][-1] in ['<', '>'] else dtype
        frame_number = int(info['frame']) + 1
        metadata = {
            'data_series': info['series'],
            'saturated_value': 1e6,
            'overloads': 0,
            'frame_number': frame_number,
            'filename': f"{self.name}-{info['series']}",
            'name': f"{self.name}-{info['series']}",
            'start_angle': self._start_angle + frame_number * self.header['delta_angle'],
        }

        try:
            if frame['encoding'].startswith('lz4'):
                arr_bytes = lz4.block.decompress(img_data, uncompressed_size=size * dtype.itemsize)
                mdata = numpy.frombuffer(arr_bytes, dtype=dtype).reshape(*shape)
            elif frame['encoding'].startswith('bs'):
                mdata = bshuf.decompress_lz4(img_data[12:], shape, dtype)
            else:
                raise RuntimeError(f'Unknown encoding {frame["encoding"]}')
            data = mdata.view(TYPES[frame['type']])

        except Exception as e:
            data = None
            logger.error(f'Error decoding stream: {e}')

        return metadata, data


def multiplexer(address, port):
    """
    Proxies the PUSH/PULL Stream from address and publishes it as a PUB/SUB Stream through port.
    This allows multiple ZMQ clients to access the same data.
    """

    try:
        context = zmq.Context()
        backend = context.socket(zmq.PULL)
        backend.connect(address)
        frontend = context.socket(zmq.PUB)
        frontend.bind(f"tcp://*:{port}")

        print(f'Starting ZMQ PULL Multiplexer proxying messages from {address} to {port} ...')
        zmq.device(zmq.STREAMER, frontend, backend)
    except Exception as e:
        pass
    finally:
        frontend.close()
        backend.close()
        context.term()

