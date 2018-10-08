from .network import Network, Subnet, Dataset
import elf
import numpy as np
import pandas as pd

def build_energy_mlcf(feature_src, target_src, masks = {}, automask_std = 0,
    filters = [], test_size = 0.2):

    if not len(feature_src) == len(target_src):
        raise Exception('Please provided only one target location for each feature set')

    sets = []
    if len(filters) != len(feature_src):
        filters = [0]*len(feature_src)

    no_mask = False
    for fsrc, tsrc, filter in zip(feature_src, target_src, filters):
        elfs = elf.utils.hdf5_to_elfs(fsrc,
                              grouped = True, values_only = True)

        targets = np.genfromtxt(tsrc, delimiter = ',')
        if not isinstance(filter, list) and not isinstance(filter, np.ndarray):
            filter = [True] * len(targets)
        if len(masks) != len(elfs):
            no_mask = True
            for species in elfs:
                feat = np.array(elfs[species])
                masks[species] = (np.std(feat.reshape(-1,feat.shape[-1]),
                        axis = 0) > automask_std)

        targets = targets[filter]
        subnets = []
        for species in elfs:
            feat = np.array(elfs[species])[:,:,masks[species]]
            feat = feat[filter]
            print(feat.shape)
            for j in range(feat.shape[1]):
                subnets.append(Subnet())
                subnets[-1].add_dataset(Dataset(feat[:,j:j+1], species),
                    targets, test_size = 0.2)

        sets.append(subnets)
    if no_mask:
        return Network(sets), masks
    else:
        return Network(sets)
