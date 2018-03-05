

import os
import tensorflow as tf
from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS
from xcml import load_network
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase import Atoms
from ase.units import Ry
from siesta_utils.mat import import_matrix
import time 
import numpy as np 

os.environ['SIESTA_COMMAND'] = 'mpirun -n 16 siesta < ./%s > ./%s'
#os.environ['SIESTA_COMMAND'] = 'siesta < ./%s > ./%s'
os.environ['QT_QPA_PLATFORM']='offscreen'
nn_path = '/gpfs/home/smdick/exchange_ml/models/final/nn_mulliken_dz/'

offset_nn = (-469.766523)

basis_sets = {'o_basis_qz' : """ 3
n=2 0 4 E 50. 7.5
    8.0 5.0 3.5 2.0
n=2 1 4 E 10. 8.3
    8.5 5.0 3.5 2.0
n=3 2 2 E 40. 8.3 Q 6.
    8.5 2.2""",

'h_basis_qz' : """ 2
n=1 0 4 E 50. 8.3
    8.5 5.0 3.5 2.0
n=2 1 2 E 20. 7.8 Q 3.5
    8.0 2.0""",

'o_basis_dz': """    3     -0.24233
n=2   0   2   E    23.36061     3.39721
     4.50769     2.64066
     1.00000     1.00000
n=2   1   2   E     2.78334     5.14253
     6.14996     2.59356
     1.00000     1.00000
n=3   2   1   E    63.98188     0.16104
     3.54403
     1.00000""",

'h_basis_dz': """2      0.46527
n=1   0   2   E    99.93138     2.59932
     4.20357     1.84463
     1.00000     1.00000
n=2   1   1   E    24.56504     2.20231
     3.52816
     1.00000"""}

class SiestaH2O(Siesta):

    def __init__(self, basis = 'qz', xc='BH'):
        os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
        species_o = Species(symbol= 'O',
         basis_set= PAOBasisBlock(basis_sets['o_basis_{}'.format(basis)]))
        species_h = Species(symbol= 'H',
         basis_set= PAOBasisBlock(basis_sets['h_basis_{}'.format(basis)]))

        super().__init__(label='H2O',
               xc=xc,
               mesh_cutoff=200 * Ry,
               species=[species_o, species_h],
               energy_shift=0.02 * Ry,
               fdf_arguments={'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 50,
                              'DM.NumberPulay': 3,
                              'DM.Tolerance': 5e-4,
                              'ElectronicTemperature': 5e-3,
                              'WriteMullikenPop': 1,
                              'DM.FormattedFiles': 'True'})
        self.nn_model = load_network(nn_path)
        
    def get_potential_energy(self, atoms, force_consistent = False):
        #TODO: Fix all magic numbers !!!
        n_mol = int(len(atoms)/3)
        pot_energy = super().get_potential_energy(atoms)
        D = import_matrix('H2O.DMF')
        S = import_matrix('H2O.S')
        DMS = D.dot(S.T)
        basis = find_basis("H2O.out")
        with open('ML_TIMES','a') as time_log:

            start = time.time()
            M = getM_from_DMS(DMS, atoms.get_positions(),
                 n_mol, basis)
            end = time.time()
            time_log.write('Time to rotate M: {} s'.format(start-end))    
        correction = use_model(M.reshape(1,-1), n_mol,
             nn=self.nn_model, n_o_orb=13, n_h_orb=5)  
        
        return pot_energy - correction - n_mol * offset_nn
