from .network import Network, Subnet, Dataset
import mlc_func.elf as elf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import keras
from sklearn.preprocessing import MinMaxScaler, Normalizer,StandardScaler
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.model_selection import train_test_split, cross_val_score
from .force_network import Force_Network
import h5py
import json

def build_force_mlcf(feature_src, target_src, mask = [], filters = [],
    automask_std = 0, autofilt_percent = 0, test_size = 0.2, species = '',
    random_state = 42):

    species = species.lower()
    if not len(species) == 1:
        raise Exception('Please specify only one species.')
    all_targets = []
    all_features = []

    if len(filters) != len(feature_src):
        filters = [0]*len(feature_src)

    basis = {}

    for fsrc, tsrc, filter in zip(feature_src, target_src, filters):
        elfs = np.array(elf.utils.hdf5_to_elfs(fsrc,
                              grouped = True, values_only = True)[species])
        angles = np.array(elf.utils.hdf5_to_elfs(fsrc,
                              grouped = True, angles_only = True)[species])

        with h5py.File(fsrc) as file:
            this_basis = json.loads(file.attrs['basis'])
            # Filter for species
            species_basis = {}
            for entry in this_basis:
                if entry[-1] == species or\
                 entry == 'alignment':
                 species_basis[entry] = this_basis[entry]

            if len(basis) > 0 and species_basis != basis:
                raise Exception('Basis used across datasets not consistent')
            else:
                basis = species_basis

        angles = angles.reshape(-1,3)

        elfs = elfs.reshape(-1,elfs.shape[-1])
        targets = np.genfromtxt(tsrc, delimiter = ',', dtype=str)
        targets = targets[targets[:,3] == species, :3].astype(float)
        print(elfs.shape)
        for idx, (t, ang) in enumerate(zip(targets, angles)):
            targets[idx] = elf.geom.rotate_vector(np.array([t]),ang.tolist(), inverse=True)

        if not len(elfs) == len(targets):
            raise Exception('Sample sizes inconsistent.')

        if not isinstance(filter, list) and not isinstance(filter, np.ndarray):
            percentile_cutoff = autofilt_percent
            selection = []
            for t in targets.T:
                lim1 = np.percentile(t, percentile_cutoff*100)
                lim2 = np.percentile(t, (1 - percentile_cutoff)*100)
                min_lim, max_lim = min(lim1,lim2), max(lim1,lim2)
                selection.append((t > min_lim) & (t < max_lim))

            filter = [s1 & s2 & s3 for s1,s2,s3 in zip(*selection)]

        if len(mask) != elfs.shape[-1]:
            feat = np.array(elfs)
            mask = (np.std(feat.reshape(-1,feat.shape[-1]),
                        axis = 0) > automask_std)
        all_features.append(elfs[:,mask][filter])
        all_targets.append(targets[filter])

    feat = np.concatenate(all_features)
    targets = np.concatenate(all_targets)

    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    X_train, X_test, y_train, y_test = train_test_split(feat,
                                                    targets,
                                                    shuffle =True,
                                                    random_state = random_state,
                                                    test_size = test_size)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                          y_train,
                                                          shuffle =True,
                                                          random_state = random_state,
                                                          test_size = 0.2)
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_test': y_test,
        'y_valid': y_valid
    }
    return Force_Network(species, scaler, basis, datasets, mask)

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
            for j in range(feat.shape[1]):
                subnets.append(Subnet())
                subnets[-1].add_dataset(Dataset(feat[:,j:j+1], species),
                    targets, test_size = 0.2)

        sets.append(subnets)
    if no_mask:
        return Network(sets), masks
    else:
        return Network(sets)
