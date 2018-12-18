import sys
import os
from mlc_func import Timer
import time
import numpy as np
import pickle
import ipyparallel as ipp
import mlc_func.elf as elf
import json

class FeatureGetter():

    def __init__(self, client = None):

        class DummyView():

            def __init__(self):
                pass
            def __len__(self):
                return 1
            def map_sync(self, *args):
                return list(map(*args))

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
    """ Reads the real space electron density and returns electronic descriptors
    """

    def __init__(self, basis, client = None, rhopath='./H2O.RHOXC'):
        """
        Parameters:
        ---
        basis: dict, dictionary defining the basis
        client: ipyparallel client for parallel processing
        rhopath: path under which the electron density can be found after every MD step
        """
        super().__init__(client = client)
        self.rhopath = rhopath
        self.basis = basis
        self.scalers = {}
        self.method = basis['alignment']
        self.masks = {}

    def set_scalers(self, scalers):
        """
        scalers: dictionary, dictionary containing the scalers that are used to transform
            the electronic descriptors before feeding them to the neural network,
            dict keys correspond to element symbols
        """
        self.scalers = scalers

    def set_masks(self, masks):
        """
        scalers: dictionary, dictionary containing the masks that are applied to
            the electronic descriptors before feeding them to the neural network,
            dict keys correspond to element symbols
        """
        self.masks = masks

    def get_features(self, atoms):

        # if not mask is set use all features
        for s in self.scalers:
            if s not in self.masks:
                self.masks[s] = [True]*1000

        density = elf.siesta.get_density_bin(self.rhopath)

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
