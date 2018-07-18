import pytest
import unittest
from elf.siesta import get_data
from elf.real_space import Density, get_elf
from ase.io import read 
import os

basis = {'r_o_o': 1.0,'r_i_o': 0.05, 'r_i_h': 0.0, 'r_o_h' : 1.5,
                      'n_rad_o' : 2,'n_rad_h' : 2, 'n_l_o' : 3,
                      'n_l_h' : 2, 'gamma_o': 20, 'gamma_h': 15}

def test_rs_elf():
    atoms = read('./test/rsd/dimer.traj')
    elf = get_elf(atoms, Density(*get_data('./test/rsd/0.RHOXC')), basis) 
    print(elf)
    
if __name__ == '__main__':
    test_rs_elf()

   
