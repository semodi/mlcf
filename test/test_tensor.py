import pytest
import unittest
from elf.siesta import get_data
from elf.real_space import Density, get_elf
from elf.geom import get_nncs_angles
from ase.io import read
import os

basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}

def test_nncs():
    atoms = read('./test/dimer.traj')
    for i in range(6):
        angles = get_nncs_angles(i, atoms.get_positions())
    
if __name__ == '__main__':
    test_nncs()
