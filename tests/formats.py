import os
import unittest

from mxio import DataSet


class DatasetTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.data_dir = os.environ.get('TEST_DATA', '/data/Xtal/tests/')

    def test_marccd_format(self):
        from mxio.formats import cbf, marccd
        dset = DataSet.new_from_file(os.path.join(self.data_dir, 'marccd', 'marccd_001.img'))
        cbf.CBFDataSet.save_frame('/tmp/marccd_0001.cbf', dset.frame)

    def test_smv_format(self):
        from mxio.formats import cbf, smv
        dset = DataSet.new_from_file(os.path.join(self.data_dir, 'smv', 'smv_1_002.img'))
        cbf.CBFDataSet.save_frame('/tmp/smv_0001.cbf', dset.frame)

    def test_cbf_format(self):
        from mxio.formats import cbf
        DataSet.new_from_file(os.path.join(self.data_dir, 'cbf', 'minicbf_0898.cbf'))

    def test_raxis_format(self):
        from mxio.formats import cbf, raxis
        dset = DataSet.new_from_file(os.path.join(self.data_dir, 'raxis', 'lysozyme_0111060520.osc'))
        cbf.CBFDataSet.save_frame('/tmp/raxis_0001.cbf', dset.frame)

    def test_hdf5_format(self):
        from mxio.formats import cbf, hdf5
        dset = DataSet.new_from_file(os.path.join(self.data_dir, 'hdf5', 'lysotest5_data_000028.h5'))
        cbf.CBFDataSet.save_frame('/tmp/hdf5_0001.cbf', dset.frame)

    def test_nexus_format(self):
        from mxio.formats import cbf, nexus
        dset = DataSet.new_from_file(os.path.join(self.data_dir, 'nexus', 'Mtb_MTR_std_pfGDR_8_2.nxs'))
        cbf.CBFDataSet.save_frame('/tmp/nexus_0001.cbf', dset.frame)


if __name__ == '__main__':
    unittest.main()