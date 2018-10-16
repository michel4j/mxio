from __future__ import print_function
from .formats import marccd, smv, raxis
from . import magic
from .common import *

_image_type_map = {
    'marCCD Area Detector Image' : marccd.MarCCDImageFile,
    'SMV Area Detector Image' : smv.SMVImageFile,
    'R-Axis Area Detector Image': raxis.RAXISImageFile,
}

try:
    from .formats import cbf
    _image_type_map['CBF Area Detector Image'] = cbf.CBFImageFile
except FormatNotAvailable:
    pass

def read_image(filename, header_only=False):
    """Determine the file type using libmagic, open the image using the correct image IO 
    back-end, and return an image object
    
    Every image Object has two attributes:
        - header: A dictionary 
        - image:  The actual PIL image of type 'I'
    """
    
    
    full_id = magic.from_file(filename).strip()
    key = full_id.split(', ')[0].strip()
    if _image_type_map.get(key) is not None:
        objClass = _image_type_map.get(key)
        #FIXME: should throw an except if img object could not be created
        img_obj = objClass(filename, header_only)
        return img_obj
    else:
        known_formats = ', '.join([v.split()[0] for v in _image_type_map.keys()])
        print('Unknown File format `%s`' % key)
        raise UnknownImageFormat('Supported formats [%s]' % (known_formats,))
        

def read_header(filename):
    img_obj = read_image(filename, header_only=True)
    return img_obj.header

__all__ = ['UnknownImageFormat', 'ImageIOError', 'read_image', 'read_header']
