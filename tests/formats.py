import os
import unittest
from pathlib import Path
from mxio import DataSet


class DatasetTestCases(unittest.TestCase):

    def setUp(self) -> None:
        from mxio.formats import cbf, raxis, hdf5, nexus, mar345, eiger, smv

        if os.path.exists('/data/Xtal/tests'):
            data_dir = Path('/data/Xtal/tests')
            self.test_data = {
                'marccd': data_dir / "marccd" / "marccd_001.img",
                'smv': data_dir / "smv" / 'smv_1_002.img',
                'cbf': data_dir / "cbf" / 'minicbf_0898.cbf',
                'raxis': data_dir / "raxis" / 'lysozyme_0111060520.osc',
                'hdf5': data_dir / "hdf5" / 'lysotest5_data_000028.h5',
                'nexus': data_dir / "nexus" / 'Mtb_MTR_std_pfGDR_8_2.nxs',
                'mar345': data_dir / "mar345" / "example.mar2300",
            }
        else:
            data_dir = Path("/home/michel/Work/Test-Data")
            self.test_data = {
                'marccd': data_dir / "marccd" / "insulin_001.img",
                'smv': data_dir / "smv" / "ALS831_lyso_Gd_010.img",
                'cbf': data_dir / "cbf" / "thaumatin_0001.cbf",
                'raxis': data_dir / "raxis" / "lysozyme_0111060100.osc",
                'hdf5': data_dir / "hdf5"/ "thau_8_master.h5",
                'nexus': data_dir / "nexus" / "xtal_1_5.nxs",
                'mar345': data_dir / "mar345" / "VTA_01_003.mar2300",
            }

    def test_marccd_format(self):
        from mxio.formats import cbf
        dset = DataSet.new_from_file(self.test_data['marccd'])
        cbf.CBFDataSet.save_frame('/tmp/marccd_0001.cbf', dset.frame)

    def test_smv_format(self):
        from mxio.formats import cbf
        dset = DataSet.new_from_file(self.test_data['smv'])
        cbf.CBFDataSet.save_frame('/tmp/smv_0001.cbf', dset.frame)

    def test_cbf_format(self):
        dset = DataSet.new_from_file(self.test_data['cbf'])

    def test_raxis_format(self):
        from mxio.formats import cbf
        dset = DataSet.new_from_file(self.test_data['raxis'])
        cbf.CBFDataSet.save_frame('/tmp/raxis_0001.cbf', dset.frame)

    def test_hdf5_format(self):
        from mxio.formats import cbf
        dset = DataSet.new_from_file(self.test_data['hdf5'])
        cbf.CBFDataSet.save_frame('/tmp/hdf5_0001.cbf', dset.frame)

    def test_nexus_format(self):
        from mxio.formats import cbf
        dset = DataSet.new_from_file(self.test_data['nexus'])
        cbf.CBFDataSet.save_frame('/tmp/nexus_0001.cbf', dset.frame)

    def test_mar345_format(self):
        from mxio.formats import cbf
        dset = DataSet.new_from_file(self.test_data['mar345'])
        cbf.CBFDataSet.save_frame('/tmp/mar345_0001.cbf', dset.frame)

if __name__ == '__main__':
    unittest.main()