import os

from .formats import marccd, smv
import magic
from .common import FormatNotAvailable, UnknownImageFormat, ImageIOError

_image_type_map = {
    'marCCD Area Detector Image': marccd.MarCCDDataSet,
    'SMV Area Detector Image': smv.SMVDataSet,
}

try:
    from .formats import cbf

    _image_type_map['CBF Area Detector Image'] = cbf.CBFDataSet
except FormatNotAvailable:
    pass

try:
    from .formats import hdf5

    _image_type_map['Hierarchical Data Format (version 5) data'] = hdf5.HDF5DataSet
except FormatNotAvailable:
    pass


def get_file_type(filename):
    m = magic.Magic(magic_file=os.path.join(os.path.dirname(__file__), 'data', 'magic'))
    return m.from_file(filename).strip()


def read_image(filename, header_only=False):
    """
    Determine the file type using libmagic, open the image using the correct image IO
    back-end, and return an image object
    
    Every image Object has two attributes:
        - header: A dictionary 
        - image:  The actual PIL image of type 'I'
    """

    full_id = get_file_type(filename)
    key = full_id.split(', ')[0].strip()
    obj_class = _image_type_map.get(key)

    if obj_class:
        img_obj = obj_class(filename, header_only)
        return img_obj
    else:
        known_formats = ', '.join([v.split()[0] for v in _image_type_map.keys()])
        raise UnknownImageFormat('Supported formats [{}]'.format(known_formats, ))


def read_header(filename):
    img_obj = read_image(filename, header_only=True)
    return img_obj.header


__all__ = ['UnknownImageFormat', 'ImageIOError', 'read_image', 'read_header']
