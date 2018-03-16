

import os
import tensorflow as tf
from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model
from xcml import load_network
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase import Atoms
from ase.units import Ry
from ase.io import Trajectory
from siesta_utils.mat import import_matrix
import time 
import numpy as np 
import pickle
#os.environ['SIESTA_COMMAND'] = 'mpirun -n 16 -f machinefile siesta < ./%s > ./%s'
os.environ['SIESTA_COMMAND'] = 'siesta < ./%s > ./%s'
os.environ['SIESTA_PP_PATH'] = '/home/sebastian/Documents/Code/siesta-4.0.1/psf/'
#os.environ['QT_QPA_PLATFORM']='offscreen'

nn_path = '/home/sebastian/Documents/Code/exchange_ml/models/nn_mulliken_dz/'
krr_path = '/home/sebastian/Documents/Code/exchange_ml/models/'

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
     3.5281
     1.00000"""}

def get_M_parallel(cwd, positions, n_mol, which):
    from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model
    from siesta_utils.mat import import_matrix
    import numpy as np

    D = import_matrix(cwd + '/H2O.DMF')
    S = import_matrix(cwd + '/H2O.S')
    DMS = D.dot(S.T)
    basis = find_basis(cwd + "/H2O.out")
    mol_list = list(range(n_mol))
    M = getM_from_DMS(DMS, positions,
                n_mol, basis, mol_list)

class SiestaH2O(Siesta):

    def __init__(self, basis = 'qz', xc='BH'):

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
                              'DM.Tolerance': 1e-4,
                              'ElectronicTemperature': 5e-3,
                              'WriteMullikenPop': 1,
                              'DM.FormattedFiles': 'True',
                              'DM.UseSaveDM': 'True'})
        self.nn_model = load_network(nn_path)

        with open(krr_path +'krr_Oxygen', 'rb') as krrfile:
            self.krr_o = pickle.load(krrfile)

        with open(krr_path +'krr_Hydrogen', 'rb') as krrfile:
            self.krr_h = pickle.load(krrfile)

        self.last_positions = None
        self.Epot = 0
        self.forces = 0 
        self.client = None
        self.view = None
        self.n_clients = 1

    def set_client(self, client):
        self.client = client
        self.view = self.client.load_balanced_view()
        self.n_clients = len(client.ids)
        if self.n_clients == 0:
            self.n_clients = 1

    def calculation_required(self, atoms, quantities = None):
        if isinstance(self.last_positions,np.ndarray):
            return not np.allclose(atoms.get_positions(), self.last_positions)
        else:
            return True

    def get_potential_energy(self, atoms, force_consistent = False):
        #TODO: Fix all magic numbers !!!
        if self.calculation_required(atoms):
            n_mol = int(len(atoms)/3)
            print(n_mol)
            time_siesta = Timer("TIME_SIESTA_BARE")
            pot_energy = super().get_potential_energy(atoms)
            forces = super().get_forces(atoms)
            time_siesta.stop()

            
            mol_list = list(range(n_mol))
            if self.n_clients > 1:
                time_getM = Timer('TIME_GETM')
                M = self.view.map_sync(get_M_parallel,[os.getcwd()]*n_mol, [atoms.get_positions()]*n_mol,
                        [n_mol]*n_mol, mol_list)
                M = np.concatenate(M)
                time_getM.stop()
            else:
                time_getM = Timer('TIME_GETM')
                time_matrix_io = Timer("TIME_MATRIX_IO")
                D = import_matrix('H2O.DMF')
                S = import_matrix('H2O.S')
                time_matrix_io.stop()
                DMS = D.dot(S.T)
                basis = find_basis("H2O.out")
                M = getM_from_DMS(DMS, atoms.get_positions(),
                         n_mol, basis)
                time_getM.stop()

            time_ML = Timer("TIME_ML")
            correction = use_model(M.reshape(1,-1), n_mol,
                 nn=self.nn_model, n_o_orb=13, n_h_orb=5)[0]
            correction_force = use_force_model(M.reshape(-1,23),self.krr_o, 
                self.krr_h,n_o_orb=13, n_h_orb=5, glob_cs= True, 
                coords = atoms.get_positions())
            time_ML.stop()
            self.last_positions = atoms.get_positions() 
            self.Epot = pot_energy - correction - n_mol * offset_nn
            self.forces = forces - correction_force.reshape(-1,3)

        return self.Epot 

    def get_forces(self, atoms):
        self.get_potential_energy(atoms)
        return self.forces
        

class Timer:

    def __init__(self, name='TIME', mode='w'):
        self.name = name
        self.start = time.time()
        self.accum = 0
        self.running = True
        self.mode = mode

    def start(self):
        self.start = time.time()
        self.running = True
    
    def pause(self):
        if self.running:
            self.accum = time.time() - self.start
            self.running = False
        else:
            raise Exception('Timer not running')
        
    def stop(self):
        if self.running:
            with open(self.name, self.mode) as timefile:
                timefile.write('{} \n'.format(time.time() - \
                    self.start+ self.accum))
            self.running = False
            self.accum = 0
        else:
            raise Exception('Timer not running')
     

def write_atoms(atoms, path, save_energy = True):
    traj = Trajectory(path,'w')
    traj.write(atoms)
    if save_energy:
        with open(path +'.energy', 'w') as efile:
            efile.write('{}\n'.format(atoms.get_potential_energy()))

def read_atoms(path, basis, xc):
    h2o = Trajectory(path,'r')[-1]
    siesta_calc = SiestaH2O(basis, xc)

    try:
        with open(path + '.energy', 'r') as efile:
            siesta_calc.Epot = float(efile.readline().strip())
        siesta_calc.last_positions = h2o.get_positions()
    except FileNotFoundError:
        print('Energy file not found. Proceeding...')

    h2o.set_calculator(siesta_calc)
    return h2o
    
 
