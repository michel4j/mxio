from mxio import DataSet


dset = DataSet.new_from_file('/users/berghuis/CMCFBM-20230802-29P9VRMK/MPHII-compound67/MPHII-compound67_6/data/MPHII-compound67_6_full_0001.cbf')
print(dset.series)
print(dset.get_sweeps())