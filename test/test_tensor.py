import pytest
import unittest
import os
import pickle
import elf.siesta as siesta
from elf.geom import make_real, make_real_old, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, tensor_to_P
from elf.real_space import Density, get_elf
from ase.io import read
import numpy as np
import xcml
basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}

def test_casimir():
    elf = pickle.load(open('test/descr_local.dat','rb'))
    # Test if rotation preserves l2-norm
    for angles in [[1.0,1.2,0.2],[0.3,3.6,2.3]]:
        rotated = rotate_tensor(elf, angles)
        casimir = get_casimir(elf)
        casimir_rotated = get_casimir(rotated)
        for key in casimir:
            assert np.allclose(casimir[key],casimir_rotated[key])
def test_identity():

    elf = pickle.load(open('test/descr_local.dat','rb'))
    # Test rotation with angles set to zero (identity transformation)
    rotated = rotate_tensor(elf, [0,0,0])
    for key in elf:
        assert np.allclose(rotated[key], elf[key])

def test_nncs():
    elf = pickle.load(open('test/descr_local.dat','rb'))
    atoms = read('./test/dimer.traj')

    # Test the NNCS alignment (nearest-neighbor rule)
    coords = atoms.get_positions().reshape(-1,3,3)[0:1]
    coords[0] -= coords[0,0]
    coords_local = np.array(coords)
    for u in range(3):
        coords_local[:,u,:] = xcml.in_local_cs(coords[:,u,:], coords.reshape(-1,3))

    angles = get_nncs_angles(0, coords_local[0])
    rotated = rotate_tensor(elf, angles, inverse = False)
    print(np.array(make_real_old(rotated)).round(4))

def test_elfcs():
    elf = pickle.load(open('test/descr_local.dat','rb'))
    atoms = read('./test/dimer.traj')
    # Establish order in which P should be  used to get ElF orientation
    P = tensor_to_P(elf)
    order = np.argsort(np.linalg.norm(P, axis = 1))
    order = order[::-1][:20]
#    order = np.arange(len(P))
#    mask = np.genfromtxt('./test/O_mask', dtype=bool)
#    order = order[mask]
    aligned = rotate_tensor(elf,
     get_elfcs_angles(elf, order, 0, atoms.get_positions()), inverse = False)
    print(np.array(make_real_old(aligned)).round(4))

if __name__ == '__main__':
    print('\n\n=======NNCS======\n\n')
    test_nncs()
    print('\n\n=======ElFCS======\n\n')
    test_elfcs()
