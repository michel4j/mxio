import cv2
import lz4.block
import lz4.frame
import numpy
import json
from ..misc import bshuf
from . import DataSet
from ..log import get_module_logger

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
    'two_theta': 'two_theta_start',
    'pixel_size': 'x_pixel_size',
    'exposure_time': 'frame_time',
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
    'distance': float,
    'beam_center': float,
    'saturated_value': int,
    'num_frames': int,
    'date': 'data_collection_date',
    'energy': float,
    'sensor_thickness': lambda v: float(v) * 1000,
    'detector_size': int,
}


class EigerStream(DataSet):
    def __init__(self):
        super().__init__()
        self._start_angle = 0.0

    def read_dataset(self):
        # dummy method, does nothing for Eiger Stream
        pass

    def read_header(self, header_data):
        header = json.loads(header_data)
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

    def read_image(self, info, series_data, img_data):
        frame = json.loads(series_data)
        dtype = numpy.dtype(frame['type'])
        shape = frame['shape'][::-1]
        size = numpy.prod(shape)
        dtype = dtype.newbyteorder(frame['encoding'][-1]) if frame['encoding'][-1] in ['<', '>'] else dtype
        frame_number = int(info['frame']) + 1
        self.header.update({
            'saturated_value': 1e6,
            'overloads': 0,
            'frame_number': frame_number,
            'filename': 'Stream',
            'name': 'Stream',
            'start_angle': self._start_angle + frame_number * self.header['delta_angle'],
        })

        try:
            if frame['encoding'].startswith('lz4'):
                arr_bytes = lz4.block.decompress(img_data, uncompressed_size=size * dtype.itemsize)
                mdata = numpy.frombuffer(arr_bytes, dtype=dtype).reshape(*shape)
            elif frame['encoding'].startswith('bs'):
                mdata = bshuf.decompress_lz4(img_data[12:], shape, dtype)
            else:
                raise RuntimeError(f'Unknown encoding {frame["encoding"]}')

            data = mdata.view(TYPES[frame['type']])
            stats_data = data[(data >= 0) & (data < self.header['saturated_value'])]
            avg, stdev = numpy.ravel(cv2.meanStdDev(stats_data))

            self.header.update({
                'average_intensity': avg,
                'std_dev': stdev,
                'min_intensity': stats_data.min(),
                'max_intensity': stats_data.max(),
            })
            self.data = data
            self.stats_data = stats_data

        except Exception as e:
            logger.error(f'Error decoding stream: {e}')


