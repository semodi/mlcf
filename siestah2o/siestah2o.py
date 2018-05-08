import sys
import os
from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model
from xcml.misc import use_model_descr, use_force_model_fd
from xcml import load_network, box_fast, fold_back_coords
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta
from timer import Timer
from feature_io import FeatureGetter, DescriptorGetter
try:
    from ase.calculators.siesta.parameters import Species
except ImportError:
    from ase.calculators.siesta.parameters import Specie as Species

from ase.calculators.siesta.parameters import PAOBasisBlock
from ase.units import Ry
from ase.io import Trajectory
import numpy as np
import pickle
import siesta_utils.grid as siesta
import ipyparallel as parallel
#os.environ['SIESTA_COMMAND'] = 'mpirun -n 16 siesta < ./%s > ./%s'
#os.environ['SIESTA_COMMAND'] = 'siesta < ./%s > ./%s'
os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
#os.environ['QT_QPA_PLATFORM']='offscreen'

#nn_path = '/home/sebastian/Documents/Code/exchange_ml/models/final/nn_mulliken_descriptors_dz/'
#krr_path = '/home/sebastian/Documents/Code/exchange_ml/models/final/'

nn_path = '/gpfs/home/smdick/exchange_ml/models/final/nn_mulliken_descriptors_dz/'
krr_path = '/gpfs/home/smdick/exchange_ml/models/final/'

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

def log_all(energy_siesta = None, energy_corrected = None,
     forces_siesta=None, forces_corrected=None, features=None):
    if energy_siesta == None: #Initialize
        with open('energies.dat', 'w') as file:
            file.write('Siesta \t Corrected \n')
        with open('forces_siesta.dat', 'w') as file:
            pass
        with open('forces_corrected.dat', 'w') as file:
            pass
        with open('features.dat', 'w') as file:
            pass
    else:
        with open('energies.dat', 'a') as file:
            file.write('{:.4f}\t{:.4f}\n'.format(energy_siesta,
                 energy_corrected))
        with open('forces_siesta.dat', 'a') as file:
            np.savetxt(file, forces_siesta, fmt = '%.4f')
        with open('forces_corrected.dat', 'a') as file:
            np.savetxt(file, forces_corrected, fmt = '%.4f')
        with open('features.dat', 'a') as file:
            np.savetxt(file, features, fmt = '%.4f')

    

class SiestaH2O(Siesta):

    def __init__(self, basis = 'qz', xc='BH', feature_getter = None, corrected = True, log_accuracy = False, use_fd = False):

        if basis != 'dz' and basis != 'qz':
            super().__init__(label='H2O',
               xc=xc,
               mesh_cutoff=200 * Ry,
               energy_shift=0.02 * Ry,
               basis_set = basis.upper())

        else: 
            species_o = Species(symbol= 'O',
             basis_set = PAOBasisBlock(basis_sets['o_basis_{}'.format(basis)]))
            species_h = Species(symbol= 'H',
             basis_set = PAOBasisBlock(basis_sets['h_basis_{}'.format(basis)]))

            super().__init__(label='H2O',
               xc=xc,
               mesh_cutoff=200 * Ry,
               species=[species_o, species_h],
               energy_shift=0.02 * Ry)

        if 'sz' in basis.lower():
            with open(krr_path +'krr_Oxygen_{}todz'.format(basis.lower()), 'rb') as krrfile:
                self.krr_o = pickle.load(krrfile)

            with open(krr_path +'krr_Hydrogen_{}todz'.format(basis.lower()),'rb') as krrfile:
                self.krr_h = pickle.load(krrfile)
        elif 'dz' in basis.lower():
            with open(krr_path +'krr_Oxygen_descr', 'rb') as krrfile:
                self.krr_o = pickle.load(krrfile)

            with open(krr_path +'krr_Hydrogen_descr', 'rb') as krrfile:
                self.krr_h = pickle.load(krrfile) 
        elif corrected:
            raise Exception('No ML model for given basis set')

        allowed_keys = self.allowed_fdf_keywords
        allowed_keys['SaveRhoXC'] = False
        self.allowed_keywords = allowed_keys
        fdf_arguments = {'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 50,
                              'DM.NumberPulay': 3,
                              'DM.Tolerance': 1e-4,
                              'ElectronicTemperature': 5e-3,
#                              'WriteMullikenPop': 1,
#                              'DM.FormattedFiles': 'True',
                              'DM.UseSaveDM': 'True',
                              'SaveRhoXC': 'True'}

        if basis != 'dz' and basis != 'qz':
            fdf_arguments['SolutionMethod'] = 'OMM'

        self.set_fdf_arguments(fdf_arguments)
        self.nn_model = load_network(nn_path)
        self.use_fd = use_fd

        with open(krr_path +'krr_dx_O_descriptors', 'rb') as krrfile:
            self.krr_o_dx = pickle.load(krrfile)

        with open(krr_path +'krr_dx_H_descriptors', 'rb') as krrfile:
            self.krr_h_dx = pickle.load(krrfile)
        

        self.last_positions = None
        self.Epot = 0
        self.forces = 0
        self.feature_getter = feature_getter
        self.corrected = corrected
        self.log_accuracy = log_accuracy
        if self.log_accuracy:
            log_all() 

    def read_eigenvalues(self):
        pass

    def read_results(self):
        """ Overrides read_results in base class to skip reading
        the charge density etc. for speed-up"""

        self.read_energy()
        self.read_forces_stress()

    def set_feature_getter(self,feature_getter):
        self.feature_getter = feature_getter

    def calculation_required(self, atoms, quantities = None):
        if isinstance(self.last_positions,np.ndarray):
            return not np.allclose(atoms.get_positions(), self.last_positions)
        else:
            return True

    def get_potential_energy(self, atoms, force_consistent = False):
        if self.calculation_required(atoms):
            self.nn_model = load_network(nn_path) # TEMP FIX 
            time_step = Timer("TIME_FULL_STEP") 
            n_mol = int(len(atoms)/3)
            time_siesta = Timer("TIME_SIESTA_BARE")
            time_siesta1 = Timer("TIME_SIESTA_1")
            pot_energy = super().get_potential_energy(atoms)
            time_siesta1.stop()
            time_siesta2 = Timer("TIME_SIESTA_2")
            forces = super().get_forces(atoms)
            time_siesta2.stop()
            time_siesta.stop()
            self.last_positions = atoms.get_positions()

            if self.corrected:
                if self.feature_getter == None:
                    raise Exception("Feature getter not defined, cannot proceed...")
                features, n_o_orb, n_h_orb =\
                    self.feature_getter.get_features(atoms.get_positions())
                n_orb = n_o_orb + 2*n_h_orb
                time_ML = Timer("TIME_ML")
                correction = use_model_descr(features.reshape(1,-1), n_mol,
                     nn=self.nn_model, n_o_orb=n_o_orb, n_h_orb=n_h_orb)[0]

                if self.use_fd:
                    correction_force = use_force_model_fd(features.reshape(-1,n_orb),self.krr_o_dx,
                        self.krr_h_dx, self.nn_model, n_o_orb=n_o_orb, n_h_orb=n_h_orb, glob_cs= True,
                        coords = fold_back_coords(atoms.get_positions(), siesta),
                        direction_factor = 1e4)
                else:
                    correction_force = use_force_model(features.reshape(-1,n_orb),self.krr_o,
                        self.krr_h, n_o_orb=n_o_orb, n_h_orb=n_h_orb, glob_cs = True,
                        coords = fold_back_coords(atoms.get_positions(), siesta))

                time_ML.stop()
                pot_energy = pot_energy - correction - n_mol * offset_nn
                forces = forces - correction_force.reshape(-1,3)
                if self.log_accuracy:
                    log_all(pot_energy + correction, pot_energy,
                         forces + correction_force.reshape(-1,3), forces,
                         features)
            self.Epot = pot_energy
            self.forces = forces 
            time_step.stop() 
        return self.Epot

    def get_forces(self, atoms):
        self.get_potential_energy(atoms)
        return self.forces


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
