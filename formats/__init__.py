

class DataSet(object):
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
