import sys
import os
from .timer import Timer
import time
import numpy as np
import pickle
import ipyparallel as parallel
import elf
import json

def find_h2o(atoms):
    atomic_numbers = atoms.get_atomic_numbers()
    indices = []
    for i, z in enumerate(atomic_numbers):
        if z == 8 and atomic_numbers[i+1] == 1 and atomic_numbers[i+2] == 1:
            indices += [i, i+1, i+2]
    return np.array(indices)

class FeatureGetter():

    def __init__(self, n_mol, n_o_orb = 13, n_h_orb = 5, client = None):

        class DummyView():

            def __init__(self):
                pass

            def map_sync(self, *args):
                return list(map(*args))

        self.n_o_orb = n_o_orb
        self.n_h_orb = n_h_orb
        self.n_mol = n_mol
        if not client == None:
            try:
                self.view = client.load_balanced_view()
                print('Clients operating : {}'.format(len(client.ids)))
                self.n_clients = len(client.ids)
            except OSError:
                print('Warning: running without ipcluster')
                self.n_clients = 0
            if self.n_clients == 0:
                print('Warning: running without ipcluster')
                self.n_clients = 1
        else:
            print('Warning: running without ipcluster')
            self.view = DummyView()
            self.n_clients = 1

class DescriptorGetter(FeatureGetter):

    def __init__(self, method, basis, client = None):
        # client = parallel.Client(profile='default')
        super().__init__(1, n_o_orb = 0, n_h_orb= 0, client = client)
        self.basis = basis
        with open('basis.json','w') as basisfile:
            basisfile.write(json.dumps(self.basis))
        self.scalers = {}
        self.method = method
        self.masks = {}
        with open('method','w') as methodfile:
            methodfile.write(self.method)
         
    def set_scalers(self, scalers):
        self.scalers = scalers

    def set_masks(self, masks):
        self.masks = masks

    def get_features(self, atoms):

        # if not mask is set use all features
        if len(self.masks) != 2:
            self.masks['o'] = [True] * 1000
            self.masks['h'] = [True] * 1000

        density = elf.siesta.get_density_bin('./H2O.RHOXC')

        elfs = elf.real_space.get_elfs_oriented(atoms, density,
                self.basis, self.method, self.view)

        elfs_dict = {}
        angles_dict = {}
        for e in elfs:
            if not e.species in elfs_dict:
                elfs_dict[e.species] = []
                angles_dict[e.species] = []
            elfs_dict[e.species].append(e.value[self.masks[e.species.lower()][:len(e.value)]])
            angles_dict[e.species].append(e.angles)

        for symbol in elfs_dict:
            try:
                elfs_dict[symbol] = self.scalers[symbol.lower()].transform(np.array(elfs_dict[symbol]))
            except KeyError:
                print("KeyError: No scaler provided for given atomic species {}".format(symbol))
                raise KeyError

        return elfs_dict, angles_dict
