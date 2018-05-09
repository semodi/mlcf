import numpy as np 
par = {}
model_basepath = '/gpfs/home/smdick/exchange_ml/models/final/'
#model_basepath = '/home/sebastian/Documents/Code/exchange_ml/models/final/'
par['descr'] = {}
par['mull'] = {}

#=============== Descriptor models ===================

#---- Energy -----
par['descr']['nn'] = {'dz_custom': 'nn_descriptors_dz_symright',
                      'qz_custom': 'nn_mulliken_symright_3_8_1e7',
                      'sz': 'nn_descriptors_sztoqz',
                      'szp': 'nn_descriptors_szp',
                      'dzp': 'nn_descriptors_dz_symright'}

#---- read_forces_stress -----
par['descr']['krr_o'] = {'dz_custom': 'krr_Oxygen_descr',
                         'sz': 'krr_Oxygen_sztodz',
                         'szp': 'krr_Oxygen_szptodz',
                         'dzp': 'krr_Oxygen_descr'}
par['descr']['krr_h'] = {}
for key in par['descr']['krr_o']:
    par['descr']['krr_h'][key] = par['descr']['krr_o'][key].replace('Oxygen','Hydrogen')

#---- Finite difference -----

par['descr']['krr_o_dx'] = {'dz_custom': 'krr_dx_O_descriptors',
                            'dzp': 'krr_dx_O_descriptors'}

par['descr']['krr_h_dx'] = {'dz_custom': 'krr_dx_H_descriptors',
                            'dzp': 'krr_dx_H_descriptors'}

#---- Symmetry factor for force model ------
par['descr']['sym'] = np.genfromtxt('symmetry.dat')

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
