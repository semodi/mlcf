import numpy as np
par = {}
model_basepath = '/gpfs/home/smdick/exchange_ml/models/final/'
#model_basepath = '/home/sebastian/Documents/Code/exchange_ml/models/final/'
par['descr'] = {}
par['mull'] = {}
par['atomic'] = {}
#=============== Descriptor models ===================

#---- Energy -----
par['descr']['nn'] = {'dz_custom': 'nn_descriptors_dz_symright',
                      'qz_custom': 'nn_mulliken_symright_3_8_1e7',
                      'sz': 'nn_descriptors_sztoqz',
                      'szp': 'nn_descriptors_szp',
                      'dzp': 'nn_descriptors_dz_symright'}

#---- Forces -----
par['descr']['krr_o'] = {'dz_custom': 'krr_Oxygen_descr',
#par['descr']['krr_o'] = {'dz_custom': 'krr_Oxygen_dz_mono',
                         'sz': 'krr_Oxygen_sztodz',
                         'szp': 'krr_Oxygen_szptodz',
                         'dzp': 'krr_Oxygen_descr',
                         'uf' : 'krr_Oxygen_descr_uftodzp'}

par['atomic']['krr_o'] = {'dz_custom': 'atom_force_O_descr'}
par['atomic']['krr_h'] = {'dz_custom': 'atom_force_H_descr'}

par['descr']['krr_h'] = {}
for key in par['descr']['krr_o']:
    par['descr']['krr_h'][key] = par['descr']['krr_o'][key].replace('Oxygen','Hydrogen')

#---- Finite difference -----

par['descr']['krr_o_dx'] = {'dz_custom': 'krr_dx_O_descriptors',
                            'dzp': 'krr_dx_O_descriptors'}

par['descr']['krr_h_dx'] = {'dz_custom': 'krr_dx_H_descriptors',
                            'dzp': 'krr_dx_H_descriptors'}

#---- Symmetry factor for force model ------

#================= Mulliken models =======================

#---- Energy -----
par['mull']['nn'] = {'dz_custom': 'nn_mulliken_dz_rand'}

#---- Forces -----
par['mull']['krr_o'] = {'dz_custom': 'krr_Oxygen_rand',
                         'szp': 'krr_Oxygen_mulliken_szptodz'}

par['mull']['krr_h'] = {}
for key in par['mull']['krr_o']:
    par['mull']['krr_h'][key] = par['mull']['krr_o'][key].replace('Oxygen','Hydrogen')

# ----- Finite difference ------
par['mull']['krr_o_dx'] = {}
par['mull']['krr_h_dx'] = {}

# ----- Orbitals --------
par['mull']['n_o_orb'] = {'dz_custom': 13,
                          'qz_custom': 26,
                          'sz': 4,
                          'szp' : 9}
par['mull']['n_h_orb'] = {'dz_custom': 5,
                          'qz_custom': 10,
                          'sz': 1,
                          'szp' : 4}
