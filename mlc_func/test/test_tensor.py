import pytest
import unittest
import os
import pickle
import elf.siesta as siesta
from elf.geom import make_real, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, tensor_to_P, rotate_vector, fold_back_coords
from elf.real_space import Density, get_elfs, orient_elfs
from ase.io import read
import numpy as np

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

def test_rotate_tensor():
    """Test certain algebraic properties of the rotate_tensor() routine
    """
    # Test identity and inverse
    id = rotate_tensor(elf[0], [0,0,0])
    rotated = rotate_tensor(elf[0], [1.2,0.3,0.1])
    rotated = rotate_tensor(rotated, [1.2,0.3,0.1], inverse = True)
    for key in elf[0]:
        assert np.allclose(rotated[key], elf[0][key])
        assert np.allclose(id[key], elf[0][key])

    # Test if rotation preserves l2-norm
    for angles in [[1.0,1.2,0.2],[0.3,3.6,2.3]]:
        rotated = rotate_tensor(elf[0], angles)
        casimir = get_casimir(elf[0])
        casimir_rotated = get_casimir(rotated)
        for key in casimir:
            assert np.allclose(casimir[key],casimir_rotated[key])

def test_rotate_vector():
    """Test certain algebraic properties of the rotate_vector() routine
    """
    vec = np.array([[1.3, 0.2, 2.1],[0,1,2],[0,0,0],[1000.123,10.222234,11]])
    ang = [0.15, 2.98, 5.16]
    # Test identity and inverse
    assert np.allclose(rotate_vector(vec,[0,0,0]),vec)
    assert np.allclose(rotate_vector(rotate_vector(vec,ang),ang,True),vec)

    assert np.allclose(np.linalg.norm(vec, axis=-1),
                       np.linalg.norm(rotate_vector(vec,ang), axis=-1))

def test_fold_back_coords():
    """ Test whether the routine fold_back_coords correctly folds the
    coordinates into the unitcell so that the distance between two points is
    minimized
    """
    coords = np.array([[1,0,1],[19,18,-20]]).astype(float)
    uc = (np.eye(3)*20).astype(float)

    coords_folded = np.array([[1,0,1],[-1,-2,0]]).astype(float)
    assert np.allclose(coords_folded,fold_back_coords(0,coords,uc))

    uc2 = np.eye(3)
    uc2[0,0] = 6
    uc2[1,1] = 10
    uc2[2,2] = 2.5

    coords_folded_2 = np.array([[1,0,1],[1,-2,0]]).astype(float)
    assert np.allclose(coords_folded_2,fold_back_coords(0,coords,uc2))

def test_p_cov():
    """ Test whether p transforms covariantly under SO(3) rotations
    """
    for i in [3,1,2,4,0]:

        p_init = tensor_to_P(elf[i])

        for it in range(10):
            rand_ang = np.random.rand(3)*2*np.pi
            elf_rotated = rotate_tensor(elf[i],rand_ang)
            p_rotated = tensor_to_P(elf_rotated)
            p = rotate_vector(p_rotated, rand_ang, inverse = True)
            assert np.allclose(p, p_init)

        for it in range(10):
            rand_ang = np.random.rand(3)*2*np.pi
            elf_rotated = rotate_tensor(elf[i],rand_ang)
            p_rotated = tensor_to_P(elf_rotated)
            p = rotate_vector(p_init, rand_ang)
            assert np.allclose(p, p_rotated)

if __name__ == '__main__':
    pass
