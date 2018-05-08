import sys
import os
from .timer import Timer
import time
import numpy as np
import pickle
import siesta_utils.grid as siesta
import ipyparallel as parallel

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

    def get_features(self, *args):
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

        time_ML = Timer("TIME_IO")
        M = find_mulliken('H2O.out', n_mol, n_o_orb= self.n_o_orb,
          n_h_orb = self.n_h_orb)
        time_ML.stop()
        
        return M, self.n_o_orb, self.n_h_orb

#    def __single_thread(self, DMS, n_mol, which_mol, cwd):
#        from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model
#        basis = find_basis(cwd + "/H2O.out")
#        M = getM_from_DMS(DMS, positions,
#                    n_mol, basis, which_mol)
#        return M

def single_thread_descriptors(coords, rho_list, grid, uc, basis):
    import os 
    os.environ['QT_QPA_PLATFORM']='offscreen'
    import xcml
    import siesta_utils.grid as siesta
    
    siesta.grid = grid
    siesta.unitcell = uc
    all_descr = []

    for c, rho, al in zip(coords,rho_list, ['o','h1','h2']):
        siesta.rho = rho
        all_descr.append(xcml.atom_decomposition(coords, siesta, basis, al))

    return np.concatenate(all_descr)

class DescriptorGetter(FeatureGetter):

    def __init__(self, client = None):
        # client = parallel.Client(profile='default')
        super().__init__(1, n_o_orb = 18, n_h_orb= 8, client = client)
        self.basis = {'r_c_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_c_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}

        self.single_thread = single_thread_descriptors

    def get_features(self, coords):
        # ========== Use if mulliken population oriented =========
        time_getfeat = Timer('TIME_GETFEAT')
        siesta.get_data_bin('./H2O.RHOXC')
        coords = coords.reshape(-1,3,3)

        rho_snippets = []
        for c in coords:
            snippet = []
            for a, l in zip(c,['o', 'h', 'h']):
                snippet.append(siesta.rho[box_fast(a,
                 self.basis['r_c_' + l], siesta)])
            rho_snippets.append(snippet)

        descr = self.view.map_sync(self.single_thread, coords, rho_snippets,
            [siesta.grid]*len(coords),[siesta.unitcell]*len(coords),
            [self.basis]*len(coords))

        time_getfeat.stop()
        descr = np.concatenate(descr)
        return descr, self.n_o_orb, self.n_h_orb


