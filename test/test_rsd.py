import pytest
import unittest
import mlc_func.elf.siesta as siesta
from mlc_func.elf.geom import make_real, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, rotate_vector
from mlc_func.elf.real_space import Density, get_elfs, orient_elfs
from ase.io import read
from mlc_func.elf.utils import preprocess_all
from mlc_func.elf.water import get_water_angles
import os
import numpy as np
import pickle

basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 0, 'gamma_h': 0}
atoms = read('./test/dimer.traj')
elf = get_elfs(atoms, siesta.get_density('./test/0.RHOXC'), basis)
elf_list = []
elf_obj = elf
for i, e in enumerate(elf):
    elf_list.append(e.value)
elf = elf_list

def test_rs_elf():
    """Test if the basic function get_elfs() works
    """
    elf = elf_list[0]
    elf_ref = pickle.load(open('./test/elf_global.dat','rb'))
    for key in elf:
        assert np.allclose(elf[key], elf_ref[key])

# def test_rot_invariance_elfcs():
#     """Test if the ElF algorithm is rotationally invariant, by calculating oriented
#     elfs for monomers that are rotated copies of each other
#     """
#     elfs = preprocess_all('./test/dimers_rotated_400', basis)
#     elfs = np.array([e[0].value for e in elfs])
#     assert np.allclose(elfs[0],elfs[1], atol = 5e-2, rtol = 5e-3)
#     assert np.allclose(elfs[1], elfs[2], atol = 5e-2, rtol = 5e-3)
#
# def test_rot_invariance_nncs():
#     """Test if the nearest-neighbor algorithm is rotationally invariant, by calculating
#     oriented elfs for monomers that are rotated copies of each other
#     """
#     elfs = preprocess_all('./test/dimers_rotated', basis, method = 'nn')
#     elfs = np.array([e[0].value for e in elfs])
#     assert np.allclose(elfs[0],elfs[1], atol = 5e-2, rtol = 5e-3)
#     assert np.allclose(elfs[1], elfs[2], atol = 5e-2, rtol = 5e-3)

def test_watercs():
    """ Test whether nearest neighbor reproduces the reference values for elf
    """
    for i in [3,1,4,0]:

        angles1 = get_water_angles(i, atoms.get_positions())
        rotated1 = rotate_tensor(elf[i], angles1, inverse = True)

        for it in range(5):
            rand_ang = np.random.rand(3)*2*np.pi
            elf_rotated = rotate_tensor(elf[i],rand_ang)
            coords_rotated = rotate_vector(atoms.get_positions()-\
                atoms.get_positions()[i], rand_ang)

            angles2 = get_water_angles(i, coords_rotated)
            rotated2 = rotate_tensor(elf_rotated, angles2, inverse = True)

            for key in rotated1:
                np.allclose(rotated1[key], rotated2[key], atol= 1e-6)
    # pickle.dump(rotated1, open('./test/elf_nncs.dat','wb'))
    # elf_ref = pickle.load(open('./test/elf_nncs.dat','rb'))
    # for key in elf[0]:
        # assert np.allclose(rotated1[key], elf_ref[key])

def test_nncs():
    """ Test whether nearest neighbor reproduces the reference values for elf
    """
    for i in [3,1,4,0]:

        angles1 = get_nncs_angles(i, atoms.get_positions())
        rotated1 = rotate_tensor(elf[i], angles1, inverse = True)

        for it in range(5):
            rand_ang = np.random.rand(3)*2*np.pi
            elf_rotated = rotate_tensor(elf[i],rand_ang)
            coords_rotated = rotate_vector(atoms.get_positions()-\
                atoms.get_positions()[i], rand_ang)

            angles2 = get_nncs_angles(i, coords_rotated)
            rotated2 = rotate_tensor(elf_rotated, angles2, inverse = True)

            for key in rotated1:
                np.allclose(rotated1[key], rotated2[key], atol= 1e-6)
    # pickle.dump(rotated1, open('./test/elf_nncs.dat','wb'))
    elf_ref = pickle.load(open('./test/elf_nncs.dat','rb'))
    for key in elf[0]:
        assert np.allclose(rotated1[key], elf_ref[key])

def test_elfcs():
    """ Test whether the ElF algorithm reproduces the reference values for elf
    """
    # Test the ElF alignment (ElF rule)
    for i in [0,3,0]:
        print('======Testing {} =========='.format(i))
        angles1 = get_elfcs_angles(i, atoms.get_positions(), elf[i])
        print(angles1)
        rotated1 = rotate_tensor(elf[i], angles1, inverse = True)

        for it in range(10):
            rand_ang = np.random.rand(3)*2*np.pi
            elf_rotated = rotate_tensor(elf[i], rand_ang)
            coords_rotated = rotate_vector(atoms.get_positions()-\
                atoms.get_positions()[i], rand_ang)

            angles2 = get_elfcs_angles(i, coords_rotated, elf_rotated)
            rotated2 = rotate_tensor(elf_rotated, angles2, inverse = True)

            for key in rotated1:
                assert np.allclose(rotated1[key], rotated2[key], atol= 1e-3, rtol = 1e-3)

    # pickle.dump(rotated1, open('./test/elf_elfcs.dat','wb'))
    elf_ref = pickle.load(open('./test/elf_elfcs.dat','rb'))
    for key in elf[0]:
        assert np.allclose(rotated1[key], elf_ref[key])

def test_orient_elfs():
    """ Test the method orient_elfs()
    """
    oriented = orient_elfs(elf_obj, atoms, mode='elf')[0].value
    elf_ref = pickle.load(open('./test/elf_elfcs.dat','rb'))
    assert np.allclose(make_real(elf_ref), oriented)

if __name__ == '__main__':
    test_elfcs()
