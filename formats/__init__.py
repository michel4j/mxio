

class DataSet(object):
    def __init__(self):
        self.header = {}

    def read_header(self):
        raise NotImplementedError('Not implemented!')

    def read_image(self):
        raise NotImplementedError('Not implemented!')

    def next_frame(self):
        raise NotImplementedError('Not implemented!')

    def prev_frame(self):
        raise NotImplementedError('Not implemented!')

    def get_frame(self, index=1):
        raise NotImplementedError('Not implemented!')