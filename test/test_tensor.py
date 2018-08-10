import pytest
import unittest
import os
import pickle
import elf.siesta as siesta
from elf.geom import make_real, make_real_old, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, tensor_to_P, rotate_vector
from elf.real_space import Density, get_elfs
from ase.io import read
import numpy as np

basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 0, 'gamma_h': 0}
atoms = read('./test/dimer.traj')
elf = get_elfs(atoms, siesta.get_density('./test/0.RHOXC'), basis)
for i, e in enumerate(elf):
    elf[i] = e.value

def test_rotate_tensor():
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

def test_nncs():
    # Test the NNCS alignment (nearest-neighbor rule)
    for i in [1,4,0]:

        angles1 = get_nncs_angles(i, atoms.get_positions())
        rotated1 = rotate_tensor(elf[i], angles1, inverse = True)

        for it in range(5):
            rand_ang = np.random.rand(3)
            elf_rotated = rotate_tensor(elf[i],rand_ang)
            coords_rotated = rotate_vector(atoms.get_positions()-\
                atoms.get_positions()[i], rand_ang)

            angles2 = get_nncs_angles(i, coords_rotated)
            rotated2 = rotate_tensor(elf_rotated, angles2, inverse = True)

            for key in rotated1:
                np.allclose(rotated1[key], rotated2[key], atol= 1e-6)

#    elf_ref = pickle.load(open('./test/elf_nncs.dat','rb'))
#    for key in elf[0]:
#        assert np.allclose(rotated1[key], elf_ref[key])

def test_elfcs():

    # Test the NNCS alignment (nearest-neighbor rule)
    for i in [0,1,4,0]:
        print(i)
        angles1 = get_elfcs_angles(i, atoms.get_positions(), elf[i])
        rotated1 = rotate_tensor(elf[i], angles1, inverse = True)

        for it in range(5):
            rand_ang = np.random.rand(3)
            elf_rotated = rotate_tensor(elf[i],rand_ang)
            coords_rotated = rotate_vector(atoms.get_positions()-\
                atoms.get_positions()[i], rand_ang)

            angles2 = get_elfcs_angles(i, coords_rotated, elf_rotated)
            rotated2 = rotate_tensor(elf_rotated, angles2, inverse = True)

            for key in rotated1:
                np.allclose(rotated1[key], rotated2[key], atol= 1e-6)

   #TODO: update reference file
    # elf_ref = pickle.load(open('./test/elf_elfcs.dat','rb'))
    # for key in elf[0]:
        # assert np.allclose(rotated1[key], elf_ref[key])


if __name__ == '__main__':
    print('\n\n=======NNCS======\n\n')
    test_nncs()
    print('\n\n=======ElFCS======\n\n')
    test_elfcs()
