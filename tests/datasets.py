import os
import unittest

from mxio import dataset


class DatasetTestCases(unittest.TestCase):

    def setUp(self) -> None:
        from mxio.plugins import raxis, cbf, hdf5, smv, marccd, nexus
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
        from mxio.plugins import cbf
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'marccd', 'marccd_001.img'))
        cbf.CBFDataSet.save_frame('/tmp/marccd_0001.cbf', dset.frame)

    def test_smv_format(self):
        from mxio.plugins import cbf
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'smv', 'smv_1_002.img'))
        cbf.CBFDataSet.save_frame('/tmp/smv_0001.cbf', dset.frame)

    def test_cbf_format(self):
        dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'cbf', 'minicbf_0898.cbf'))

    def test_raxis_format(self):
        from mxio.plugins import cbf
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'raxis', 'lysozyme_0111060520.osc'))
        cbf.CBFDataSet.save_frame('/tmp/raxis_0001.cbf', dset.frame)

    def test_hdf5_format(self):
        from mxio.plugins import cbf
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'hdf5', 'lysotest5_data_000028.h5'))
        cbf.CBFDataSet.save_frame('/tmp/hdf5_0001.cbf', dset.frame)

    def test_nexus_format(self):
        from mxio.plugins import cbf
        dset = dataset.DataSet.new_from_file(os.path.join(self.data_dir, 'nexus', 'Mtb_MTR_std_pfGDR_8_2.nxs'))
        cbf.CBFDataSet.save_frame('/tmp/nexus_0001.cbf', dset.frame)


if __name__ == '__main__':
    unittest.main()