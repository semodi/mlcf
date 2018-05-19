import sys
import os
from xcml.misc import use_model, find_mulliken, getM_, find_basis, getM_from_DMS, use_force_model
from xcml.misc import use_model_descr, use_force_model_fd
from xcml import load_network, box_fast, fold_back_coords
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta
from .timer import Timer
from .feature_io import FeatureGetter, DescriptorGetter, MullikenGetter
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
from siesta_utils.grid import AtoBohr
import ipyparallel as parallel

offset_nn = (-469.766523)

basis_sets = {'o_basis_qz_custom' : """ 3
n=2 0 4 E 50. 7.5
    8.0 5.0 3.5 2.0
n=2 1 4 E 10. 8.3
    8.5 5.0 3.5 2.0
n=3 2 2 E 40. 8.3 Q 6.
    8.5 2.2""",

'h_basis_qz_custom' : """ 2
n=1 0 4 E 50. 8.3
    8.5 5.0 3.5 2.0
n=2 1 2 E 20. 7.8 Q 3.5
    8.0 2.0""",

'o_basis_dz_custom': """    3     -0.24233
n=2   0   2   E    23.36061     3.39721
     4.50769     2.64066
     1.00000     1.00000
n=2   1   2   E     2.78334     5.14253
     6.14996     2.59356
     1.00000     1.00000
n=3   2   1   E    63.98188     0.16104
     3.54403
     1.00000""",

'h_basis_dz_custom': """2      0.46527
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

    def __init__(self, basis = 'qz', xc='BH', feature_getter = None, log_accuracy = False):

        fdf_arguments = {'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 50,
                              'DM.NumberPulay': 3,
                              'ElectronicTemperature': 5e-3,
                              'WriteMullikenPop': 1,
#                              'DM.FormattedFiles': 'True',
                              'MaxSCFIterations': 20,
                              'SaveRhoXC': 'True'}


        if basis == 'uf':
            super().__init__(label='H2O',
               xc='PBE',
               mesh_cutoff=100 * Ry,
               energy_shift=0.02 * Ry,
               basis_set = 'SZ')
            dmtol = 5e-4
        
        elif not 'custom' in basis.lower():
            super().__init__(label='H2O',
               xc=xc,
               mesh_cutoff=200 * Ry,
               energy_shift=0.02 * Ry,
               basis_set = basis.upper())
            dmtol = 1e-4
        else:
            species_o = Species(symbol= 'O',
             basis_set = PAOBasisBlock(basis_sets['o_basis_{}'.format(basis)]))
            species_h = Species(symbol= 'H',
             basis_set = PAOBasisBlock(basis_sets['h_basis_{}'.format(basis)]))

            super().__init__(label='H2O',
               xc=xc,
               mesh_cutoff=200 * Ry,
               basis_set = 'DZP',
               species=[species_o, species_h],
               energy_shift=0.02 * Ry)
            dmtol = 1e-4
            fdf_arguments['PAO.SplitNorm'] = 0.15
            fdf_arguments['PAO.SoftDefault'] =  'True'
            fdf_arguments['PAO.SoftInnerRadius'] =  0.9
            fdf_arguments['PAO.SoftPotential'] = 1

        fdf_arguments['DM.UseSaveDM'] = 'True'
        fdf_arguments['DM.Tolerance'] = dmtol

        allowed_keys = self.allowed_fdf_keywords
        allowed_keys['SaveRhoXC'] = False
        self.allowed_keywords = allowed_keys
        self.krr_o = None
        self.krr_h = None
        self.krr_o_dx = None
        self.krr_h_dx = None
        self.set_fdf_arguments(fdf_arguments)
        self.nn_path = ''
        self.corrected_e = False
        self.corrected_f = False
        self.use_fd = False
        self.last_positions = None
        self.Epot = 0
        self.forces = 0
        self.feature_getter = feature_getter
        self.log_accuracy = log_accuracy
        self.symmetry = 0
        if self.log_accuracy:
            log_all()

    def set_solution_method(self, method):
        if not method.lower() == 'diagon' and not method.upper() == 'OMM':
            raise Exception('Invalid solution method: choose "diagon" or "OMM"')
        else:
            fdf_arguments = self.parameters['fdf_arguments']
            fdf_arguments['SolutionMethod'] = method
            self.set_fdf_arguments(fdf_arguments)

    def set_nn_path(self, path):
        self.nn_path = path
        load_network(self.nn_path)
        self.corrected_e = True

    def set_force_model(self, krr_o, krr_h):
        self.krr_o = krr_o
        self.krr_h = krr_h
        self.corrected_f = True

    def set_fd_model(self, krr_o_dx, krr_h_dx):
        self.krr_o_dx = krr_o_dx
        self.krr_h_dx = krr_h_dx
        self.use_fd = True
        self.corrected_f = True

    def set_symmetry(self, sym):
        if not isinstance(sym, np.ndarray):
            raise Exception('sym must be np.ndarray')
        else:
            self.symmetry = sym

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
#            atoms.set_positions(atoms.get_positions(wrap = True))

            time_step = Timer("TIME_FULL_STEP")
            n_mol = int(len(atoms)/3)
            time_siesta = Timer("TIME_SIESTA_BARE")
            pot_energy = super().get_potential_energy(atoms)
            forces = super().get_forces(atoms)
            time_siesta.stop()
            self.last_positions = atoms.get_positions()

            if self.corrected_e or self.corrected_f:

                if self.feature_getter == None:
                    raise Exception("Feature getter not defined, cannot proceed...")
                features, n_o_orb, n_h_orb, h2o_indices =\
                    self.feature_getter.get_features(atoms)
                n_orb = n_o_orb + 2*n_h_orb
                time_ML = Timer("TIME_ML")

                if self.corrected_e:
                    self.nn_model = load_network(self.nn_path) # TEMP FIX
                    correction = use_model_descr(features.reshape(1,-1), n_mol,
                         nn=self.nn_model, n_o_orb=n_o_orb, n_h_orb=n_h_orb, sym = self.symmetry)[0]
                else:
                    correction = 0
                siesta.unitcell = np.zeros([3,3])
                siesta.unitcell[0, 0] = atoms.get_cell()[0, 0]*AtoBohr #TEMP FIX

                if self.use_fd:
                    if not self.corrected_e:
                        raise Exception('No energy model provided for finite difference')
                    correction_force = use_force_model_fd(features.reshape(-1,n_orb),self.krr_o_dx,
                        self.krr_h_dx, self.nn_model, n_o_orb=n_o_orb, n_h_orb=n_h_orb, glob_cs= True,
                        coords = fold_back_coords(atoms.get_positions()[h2o_indices], siesta),
                        direction_factor = 1e4, sym = self.symmetry)
                elif self.corrected_f:
                    correction_force = use_force_model(features.reshape(-1,n_orb),self.krr_o,
                        self.krr_h, n_o_orb=n_o_orb, n_h_orb=n_h_orb, glob_cs = True,
                        coords = fold_back_coords(atoms.get_positions()[h2o_indices], siesta), sym = self.symmetry)
                else:
                    correction_force = np.zeros_like(forces)

                time_ML.stop()
                pot_energy = pot_energy - correction - n_mol * offset_nn
                forces[h2o_indices] = forces[h2o_indices] - correction_force.reshape(-1,3)
                if self.log_accuracy:
                    forces_uncorrected = np.array(forces)
                    forces_uncorrected[h2o_indices] += correction_force.reshape(-1,3)
                    log_all(pot_energy + correction, pot_energy,
                         forces_uncorrected, forces,
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
