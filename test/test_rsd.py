import pytest
import unittest
import elf.siesta as siesta
from elf.geom import make_real, make_real_old, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, tensor_to_P
from elf.real_space import Density, get_elfs
from ase.io import read
from elf.utils import preprocess_all
import os
import numpy as np
import pickle
# basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      # 'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      # 'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}
basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 0, 'gamma_h': 0}
atoms = read('./test/dimer.traj')
elf = get_elfs(atoms, siesta.get_density('./test/0.RHOXC'), basis)
for i, e in enumerate(elf):
    elf[i] = e.value
elf = elf[0]

def test_rs_elf():
    i = 0
    # Read test geometry + density (dimer)
    elf_ref = pickle.load(open('./test/elf_global.dat','rb'))
    for key in elf:
        assert np.allclose(elf[key], elf_ref[key])

def test_rot_invariance_elfcs():
    elfs = preprocess_all('./test/monomers_rotated', basis)
    elfs = np.array([e[0].value for e in elfs])
    assert np.allclose(elfs[0],elfs[1], atol = 5e-2, rtol = 5e-3)
    assert np.allclose(elfs[1], elfs[2], atol = 5e-2, rtol = 5e-3)

def test_rot_invariance_nncs():
    elfs = preprocess_all('./test/monomers_rotated', basis,method = 'nn')
    elfs = np.array([e[0].value for e in elfs])
    assert np.allclose(elfs[0],elfs[1], atol = 5e-2, rtol = 5e-3)
    assert np.allclose(elfs[1], elfs[2], atol = 5e-2, rtol = 5e-3)

if __name__ == '__main__':
    test_rot_invariance_elfcs()
