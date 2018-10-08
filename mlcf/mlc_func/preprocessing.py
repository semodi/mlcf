""" Module that contains functions to preprocess quantumm chemical data such as
atomic positions and/or positions and width of Gaussians that are fitted
to the charge density. Preprocessed datasets are then used in subnet and network,
for a Neural Network similar to that proposed by Behler et al.
"""

import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# Dataset containing the coordinates and width ( = 1 for atoms) for
# Gaussians and Atoms
Dataset = namedtuple("Dataset", "data species n")

def reshape_group(x, n):
    """Reshape data from format (n_samples, n_copies * 4)
    into format (n_copies, n_samples, 4) needed by tensorflow
    """

    n0 = x.shape[0]
    n1 = int(x.shape[1]/n)
    x = x.T.reshape(n,n1,n0).swapaxes(1,2)

    return x

# def remove_outliers(all_subnets, which_set='train', use='features'):
#
#
#     subnet_dict = {}
#
#     def add_to_dict(sn):
#         if not sn.species in subnet_dict:
#             subnet_dict[sn.species] = []
#         subnet_dict[sn.species].append(sn)
#
#     for sn1 in all_subnets:
#         if isinstance(sn1, list):
#             for sn1 in sn1:
#                 add_to_dict(sn1)
#         else:
#             add_to_dict(sn1)
#
#     for species in subnet_dict:
#         subnets = subnet_dict[species]
#         for s in subnets:
#             if not s.species == subnets[0].species:
#                 raise Exception('Subnets do not contain the same species. Proceeding...')
#
#         if not isinstance(subnets ,list):
#             Exception('Input must be a list of subnets')
#         if which_set == 'train':
#             all_feat = np.concatenate([s.scaler.inverse_transform(s.X_train.reshape(-1, s.features)) for s in subnets])
#             all_tar = np.concatenate([s.scaler.inverse_transform(s.y_train) for s in subnets])
#         else:
#             all_feat = np.concatenate([s.scaler.inverse_transform(s.X_test.reshape(-1, s.features)) for s in subnets])
#             all_tar = np.concatenate([s.scaler.inverse_transform(s.y_test) for s in subnets])
#
def scale_together(all_subnets):
    """ Scale data in given subnets by combining the data ranges
        (Should always be performed if subnets share weights)
    """
    print('scale_together')
    subnet_dict = {}

    def add_to_dict(sn):
        if not sn.species in subnet_dict:
            subnet_dict[sn.species] = []
        subnet_dict[sn.species].append(sn)

    for sn1 in all_subnets:
        if isinstance(sn1, list):
            for sn1 in sn1:
                add_to_dict(sn1)
        else:
            add_to_dict(sn1)
    for species in subnet_dict:
        subnets = subnet_dict[species]
        for s in subnets:
            if not s.species == subnets[0].species:
                raise Exception('Subnets do not contain the same species. Proceeding...')

        if not isinstance(subnets ,list):
            Exception('Input must be a list of subnets')

        all_data = np.concatenate([s.scaler.inverse_transform(s.X_train.reshape(-1, s.features)) for s in subnets])
        # new_scaler = MinMaxScaler(feature_range = (-2,2))
        new_scaler = StandardScaler()
        new_scaler.fit(all_data)
        for i,s in enumerate(subnets):
            subnets[i].X_train = s.scaler.inverse_transform(s.X_train.reshape(-1,s.features)).reshape(s.X_train.shape)
            subnets[i].X_test = s.scaler.inverse_transform(s.X_test.reshape(-1,s.features)).reshape(s.X_test.shape)
            subnets[i].scaler = new_scaler
            subnets[i].X_train = s.scaler.transform(subnets[i].X_train.reshape(-1,s.features)).reshape(s.X_train.shape)
            subnets[i].X_test = s.scaler.transform(subnets[i].X_test.reshape(-1,s.features)).reshape(s.X_test.shape)
