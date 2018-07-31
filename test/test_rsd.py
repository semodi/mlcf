import pytest
import unittest
import elf.siesta as siesta
from elf.geom import make_real, make_real_old, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, tensor_to_P
from elf.real_space import Density, get_elf
from ase.io import read
import os
import numpy as np
import xcml
import pickle
# basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      # 'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      # 'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}
basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 0, 'gamma_h': 0}
def test_rs_elf():
    i = 0
    # Read test geometry + density (dimer)
    atoms = read('./test/dimer.traj')
    elf = get_elf(atoms, Density(*siesta.get_data('./test/0.RHOXC')), basis)[0]
    elf_ref = pickle.load(open('./test/descr_global.dat','rb'))

    for key in elf:
        assert np.allclose(elf[key], elf_ref[key])

if __name__ == '__main__':
    test_rs_elf()
