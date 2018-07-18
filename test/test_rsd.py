import pytest
import unittest
from elf.siesta import get_data
from elf.geom import make_real, rotate_tensor, get_nncs_angles, get_casimir
from elf.real_space import Density, get_elf
from ase.io import read
import os
import numpy as np

basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}

def test_rs_elf():
    atoms = read('./test/dimer.traj')
    elf = get_elf(atoms, Density(*get_data('./test/0.RHOXC')), basis)

    angles = get_nncs_angles(0, atoms.get_positions())
    rotated = rotate_tensor(elf[0], [0,0,0])
    for key in elf[0]:
        assert np.allclose(rotated[key], elf[0][key])
    rotated = rotate_tensor(elf[0], angles)
    casimir = get_casimir(elf[0])
    casimir_rotated = get_casimir(rotated)
    print(type(elf[0]['0,0,0']))
    for key in casimir:
            assert np.allclose(casimir[key],casimir_rotated[key])

if __name__ == '__main__':
    test_rs_elf()
