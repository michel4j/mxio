import os
import unittest
from pathlib import Path
from mxio import dataset


class DatasetTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = os.environ.get('TEST_DATA', '/data/Xtal/tests/')

    def test_summarize_range(self):
        summarized = dataset.summarize_sequence([1, 2, 3, 4])
        expected = [(1, 4)]
        self.assertEqual(summarized, expected, f'Range not summarized {summarized=} != {expected=}')

    def test_summarize_skips(self):
        summarized = dataset.summarize_sequence([1, 2, 3, 5, 6, 7])
        expected = [(1, 3), (5, 7)]
        self.assertEqual(summarized, expected, f'Range not summarized {summarized=} != {expected=}')

    def test_summarize_singles(self):
        summarized = dataset.summarize_sequence([1, 5, 7])
        expected = [(1, 1), (5, 5), (7, 7)]
        self.assertEqual(summarized, expected, f'Range not summarized {summarized=} != {expected=}')

    def test_marccd_format(self):
        from mxio.plugins import marccd
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'marccd', 'marccd_001.img'))

    def test_cbf_format(self):
        from mxio.plugins import cbf
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'cbf', 'minicbf_0898.cbf'))

    def test_cbf_save(self):
        from mxio.plugins import cbf, marccd
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'marccd', 'marccd_001.img'))
        cbf.CBFDataSet.save_frame('/tmp/test_0001.cbf', dset.frame)

if __name__ == '__main__':
    unittest.main()