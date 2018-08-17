import sys
import os
from .timer import Timer
from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model, find_mulliken_h2o_indices
import time
from xcml import load_network, box_fast, fold_back_coords, rotate_vector_real
import numpy as np
import pickle
import siesta_utils.grid as siesta
import ipyparallel as parallel
import elf


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

class MullikenGetter(FeatureGetter):

    def __init__(self, n_mol, client = None):
        # client = parallel.Client(profile='default')
        super().__init__(n_mol, n_o_orb = 13, n_h_orb= 5, client = client)
        self.n_o_orb = 13

    def get_features(self, atoms, *args):
#
#        # ========== Use if mulliken population oriented =========
#
#        time_getM = Timer('TIME_GETM')
#        time_matrix_io = Timer("TIME_MATRIX_IO")
#        D = import_matrix('H2O.DMF')
#        S = import_matrix('H2O.S')
#        time_matrix_io.stop()
#        DMS = D.dot(S.T)
#        basis = find_basis("H2O.out")
#
#        M = self.view.map_sync(__single_thread, [DMS]*n_mol, [n_mol]*n_mol,
#            list(range(n_mol)),[os.getcwd()]*n_mol)
#        M = np.concatenate(M, axis = 1)
#        time_getM.stop()
#
        # ========== Use if mulliken population non-oriented ======
        h2o_indices = find_h2o(atoms)
        time_ML = Timer("TIME_IO")
        M = find_mulliken_h2o_indices('H2O.out', self.n_mol, n_o_orb= self.n_o_orb,
          n_h_orb = self.n_h_orb, h2o_indices = h2o_indices)
        time_ML.stop()

        return M, self.n_o_orb, self.n_h_orb, h2o_indices

#    def __single_thread(self, DMS, n_mol, which_mol, cwd):
#        from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model
#        basis = find_basis(cwd + "/H2O.out")
#        M = getM_from_DMS(DMS, positions,
#                    n_mol, basis, which_mol)
#        return M

class DescriptorGetter(FeatureGetter):

    def __init__(self, client = None):
        # client = parallel.Client(profile='default')
        super().__init__(1, n_o_orb = 0, n_h_orb= 0, client = client)
        self.basis = {'r_c_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_c_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 0, 'gamma_h': 0}

        self.single_thread = single_thread_descriptors_molecular
        self.scalers = {}

    def set_scalers(self, scalers):
        self.scalers = scalers

    def get_features(self, atoms):
        density = elf.siesta.get_density_bin('./H2O.RHOXC')

        elfs = elf.real_space.get_elfs_oriented(atoms, density,
                self.basis, self.view)

        elfs_dict = {}
        angles_dict = {}
        for e in elfs:
            if not e.species in elfs_dict:
                elfs_dict[e.species] = []
                angles_dict[e.species] = []
            elfs_dict[e.species].append(e.value)
            angles_dict[e.species].append(e.angles)

        for symbol in elfs_dict:
            try:
                elfs_dict[symbol] = self.scalers[symbol].transform(np.array(elfs_dict[symbol]))
            except KeyError:
                print("KeyError: No scaler provided for given atomic species {}".format(symbol))

        return elfs_dict, angles_dict
