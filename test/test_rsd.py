import pytest
import unittest
import mlc_func.elf.siesta as siesta
from mlc_func.elf.geom import make_real, rotate_tensor, get_nncs_angles,\
 get_casimir, get_elfcs_angles, rotate_vector, fold_back_coords
from mlc_func.elf.real_space import Density, get_elfs, orient_elfs
from ase.io import read
from mlc_func.elf.utils import preprocess_all, hdf5_to_elfs_fast, hdf5_to_elfs, elfs_to_hdf5, change_alignment
from mlc_func.elf.water import get_water_angles, waterc_to_tip4p, tip4p_to_str
import os
import numpy as np
import pickle
from ase.io import read

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


def test_water():

    coords = atoms.get_positions()
    tip4p = waterc_to_tip4p(coords)
    tip4p_str = tip4p_to_str(tip4p)

    # pickle.dump({'tip4p': tip4p, 'str': tip4p_str}, open('water_ref.pckl','wb'))
    ref = pickle.load(open('./test/water_ref.pckl','rb'))
    assert np.allclose(ref['tip4p'],tip4p)
    assert tip4p_str == ref['str']
def test_elf_utils():

    preprocess_all('./test/', basis = basis)

    def compare_csv(file1,file2):
        assert np.allclose(np.genfromtxt(file1, delimiter = ','),
                            np.genfromtxt(file2, delimiter = ','))


    compare_csv('./test.energies', './test/util_ref/test.energies')
    compare_csv('./test.forces', './test/util_ref/test.forces')
    assert read('./test.traj',':') == read('./test/util_ref/test.traj',':')

    def compare_elfs(file1, file2):
        for i in range(2):
            for spec in ['o','h']:
                assert np.allclose(hdf5_to_elfs_fast(file1)[i][spec],
                 hdf5_to_elfs_fast(file2)[i][spec])

    compare_elfs('./test_processed.hdf5','./test/util_ref/test_processed.hdf5')

    for spec in ['o','h']:
        assert np.allclose(hdf5_to_elfs_fast('./test_processed.hdf5')[0][spec],
         hdf5_to_elfs('./test_processed.hdf5',grouped=True, values_only=True)[spec])


    reference_elfs = hdf5_to_elfs('./test/util_ref/test_processed.hdf5')
    elfs_to_hdf5(reference_elfs, './test_saved.hdf5')


    compare_elfs('./test_saved.hdf5','./test/util_ref/test_processed.hdf5')


    preprocess_all('./test/', basis = basis, method = 'nn')
    change_alignment('./test_processed.hdf5', './test.traj', new_method = 'elf',
        save_as = './test_change_alignment.hdf5')

    compare_elfs('./test_change_alignment.hdf5','./test/util_ref/test_processed.hdf5')
    
    os.remove('./test_change_alignment.hdf5')
    os.remove('./test_processed.hdf5')
    os.remove('./test.energies')
    os.remove('./test.forces')
    os.remove('./test.traj')
    os.remove('./test_saved.hdf5')

def test_siesta():
    """ Test all routines that import data from SIESTA files
    """

    energy = siesta.get_energy('./test/0.out')
    forces = siesta.get_forces('./test/0.out')

    # pickle.dump({'energy': energy, 'forces': forces, 'atoms': atoms},
     # open('./test/siesta_test.pckl','wb'))
    ref = pickle.load(open('./test/siesta_test.pckl','rb'))
    assert energy == ref['energy']
    assert np.allclose(forces,ref['forces'])

def test_rs_elf():
    """Test if the basic function get_elfs() works
    """
    elf = elf_list[0]
    elf_ref = pickle.load(open('./test/elf_global.dat','rb'))
    for key in elf:
        assert np.allclose(elf[key], elf_ref[key])


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

if __name__ == '__main__':
    pass
