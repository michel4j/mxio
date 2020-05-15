import glob
import importlib
import os
from functools import lru_cache

from .. import log

logger = log.get_module_logger(__name__)


class DataSet(object):
    name = 'Magic Name'

    def __init__(self):
        self.header = {}
        self.name = ''
        self.data = None
        self.stats_data = None

    def read_dataset(self):
        raise NotImplementedError('Not implemented!')

    def next_frame(self):
        raise NotImplementedError('Not implemented!')

    def prev_frame(self):
        raise NotImplementedError('Not implemented!')

    def get_frame(self, index=1):
        raise NotImplementedError('Not implemented!')


@lru_cache(maxsize=None)
def get_formats():
    pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.py')
    modules = [
        os.path.splitext(os.path.basename(mod))[0]
        for mod in glob.glob(pattern) if not mod.endswith('__init__.py')
    ]

    for module in modules:
        try:
            importlib.import_module(f'.{module}', __name__)
        except (ImportError, TypeError) as err:
            print(err)
            logger.error('Format "{}" not available'.format(module))

    return {
        cls.name: cls
        for cls in DataSet.__subclasses__()
    }