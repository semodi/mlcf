import sys
import os
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta
from mlc_func import Timer
from .feature_io import FeatureGetter, DescriptorGetter
import keras
try:
    from ase.calculators.siesta.parameters import Species
except ImportError:
    from ase.calculators.siesta.parameters import Specie as Species #fix
try:
    from mbpol_calculator import MbpolCalculator
    MBPOL_AVAIL = True
except ImportError:
    MBPOL_AVAIL = False

from ase.calculators.siesta.parameters import PAOBasisBlock
from ase.units import Ry
from ase.io import Trajectory
from ase import Atoms
import numpy as np
import pickle
import ipyparallel as ipp
import mlc_func.elf as elf
import json
import subprocess
from .read_input import read_input
from mlc_func.ml import load_force_model
from .mixer import Mixer
from .siesta_basis import basis_sets

def log_all(energy_siesta = None, energy_corrected = None,
     forces_siesta=None, forces_corrected=None, features=None):
    if energy_siesta == None: #Initialize
        with open('energies.dat', 'w') as file:
            file.write('Siesta \t Corrected \n')
        with open('forces_siesta.dat', 'w') as file:
            pass
        with open('forces_corrected.dat', 'w') as file:
            pass
    else:
        with open('energies.dat', 'a') as file:
            file.write('{:.4f}\t{:.4f}\n'.format(energy_siesta,
                 energy_corrected))
        with open('forces_siesta.dat', 'a') as file:
            np.savetxt(file, forces_siesta, fmt = '%.4f')
        with open('forces_corrected.dat', 'a') as file:
            np.savetxt(file, forces_corrected, fmt = '%.4f')
        for key in features:
            with open('features_{}.dat'.format(key), 'a') as file:
                np.savetxt(file, features[key], fmt = '%.4f')

class Siesta_Calculator(Siesta):
    """ Provides default option for the Siesta calculator
    """

    def __init__(self, basis = 'qz', xc='BH'):

        label = 'H2O' #TODO: for now, change later
        if xc =='REVPBE': xc = 'revPBE'

        fdf_arguments = {'DM.MixingWeight': 0.3,
                          'DM.NumberPulay': 3,
                          'ElectronicTemperature': 5e-3,
                          'WriteMullikenPop': 0,
                          'MaxSCFIterations': 20,
                          'SaveRhoXC': 'True'}

        if basis == 'uf':
            super().__init__(label=label,
               xc='PBE',
               mesh_cutoff=100 * Ry,
               energy_shift=0.02 * Ry,
               basis_set = 'SZ')
            dmtol = 5e-4

        elif not 'custom' in basis.lower():
            super().__init__(label=label,
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

        fdf_arguments['DM.UseSaveDM'] = 'True'
        fdf_arguments['DM.Tolerance'] = dmtol

        allowed_keys = self.allowed_fdf_keywords
        allowed_keys['SaveRhoXC'] = False
        self.allowed_keywords = allowed_keys
        self.set_fdf_arguments(fdf_arguments)


    def set_solution_method(self, method):
        if not method.lower() == 'diagon' and not method.upper() == 'OMM':
            raise Exception('Invalid solution method: choose "diagon" or "OMM"')
        else:
            fdf_arguments = self.parameters['fdf_arguments']
            fdf_arguments['SolutionMethod'] = method
            self.set_fdf_arguments(fdf_arguments)

    def read_eigenvalues(self):
        pass

    def read_results(self):
        """ Overrides read_results in base class to skip reading
        the charge density etc. for speed-up"""

        self.read_energy()
        self.read_forces_stress()

class MLCF_Calculator:

    def __init__(self, base_calculator = None, feature_getter = None,
                    log_accuracy = True):

        self.force_models = {}
        self.last_positions = None
        self.Epot = 0
        self.forces = 0
        self.feature_getter = feature_getter
        self.log_accuracy = log_accuracy
        self.base_calculator = base_calculator
        self.cm_corrected = False
        if self.log_accuracy:
            log_all()

    def set_force_model(self, models):
        self.force_models = models

    def set_feature_getter(self,feature_getter):
        self.feature_getter = feature_getter

    def calculation_required(self, atoms, quantities = None):
        if isinstance(self.last_positions,np.ndarray):
            return not np.allclose(atoms.get_positions(), self.last_positions)
        else:
            return True

    def set_base_calculator(self, base_calculator):
        self.base_calculator = base_calculator
    def get_potential_energy(self, atoms, force_consistent = False):
        if self.calculation_required(atoms):
            time_step = Timer("TIME_FULL_STEP")
            time_siesta = Timer("TIME_SIESTA_BARE")
            # Use the base calculator for a first approximation to energy
            # and forces
            pot_energy = self.base_calculator.get_potential_energy(atoms)
            forces = self.base_calculator.get_forces(atoms)
            correction_force = np.zeros_like(forces)
            time_siesta.stop()
            self.last_positions = atoms.get_positions()

            if len(self.force_models) > 0:
                print('Using force correction')
                time_ML = Timer("TIME_ML")
                if self.feature_getter == None:
                    raise Exception("Feature getter not defined, cannot proceed...")

                time_feat = Timer('TIME_FEAT')
                elfs_dict, angles_dict =\
                    self.feature_getter.get_features(atoms)
                time_feat.stop()

                prediction = {}
                time_predict = Timer("TIME_PREDICT")
                for species in elfs_dict:
                    prediction[species] = self.force_models[species.lower()].predict(elfs_dict[species],
                     processed = True)
                    for i, (pred, e, a) in enumerate(zip(prediction[species],
                        elfs_dict[species], angles_dict[species])):

                        prediction[species][i] = elf.geom.rotate_vector(np.array([pred]),
                                                                a, False)
                time_predict.stop()

                time_ML.stop()
                for key in prediction:
                    prediction[key] = prediction[key].tolist()

                masses = []
                species_counter = {}
                for i, chem_sym in enumerate(atoms.get_chemical_symbols()):
                    if chem_sym in prediction:
                        correction_force[i] = np.array(prediction[chem_sym].pop(0))
                        masses.append(atoms.get_masses()[i])
                    else:
                        masses.append(0)

                masses = np.array(masses).reshape(-1,1)

                for key in prediction:
                    assert len(prediction[key]) == 0

                if self.cm_corrected:
                    # Subtract mean force correction to fix the center of mass
                    mean_correction = np.mean(correction_force, axis = 0)*len(correction_force)/np.sum(masses)
                    correction_force -= mean_correction * masses

                features = {}
                for key in elfs_dict:
                    features[key] = np.concatenate([np.array(elfs_dict[key]),
                                                    np.array(angles_dict[key])],
                                                    axis = -1)
            else:
                features = {}
            forces = forces + correction_force.reshape(-1,3)

            if self.log_accuracy:
                forces_uncorrected = np.array(forces)
                forces_uncorrected -= correction_force.reshape(-1,3)
                log_all(pot_energy, pot_energy,
                     forces_uncorrected, forces,
                     features)
            self.Epot = pot_energy
            self.forces = forces
            time_step.stop()
            subprocess.call('rm fdf*', shell = True)
            subprocess.call('rm INPUT*', shell = True)
        return self.Epot

    def get_forces(self, atoms):
        self.get_potential_energy(atoms)
        return self.forces

def load_from_file(input_file):
    settings, mixing_settings = read_input(input_file)

    if settings['mixing']:
        iterate_over = ['1','2']
        settings_choice = mixing_settings
    else:
        iterate_over = ['']
        settings_choice = settings
    calculators = []

    for i in iterate_over:
        model_path = settings_choice['model' + i]
        if not model_path == None and 'mbpol' in  model_path.lower():
            if MBPOL_AVAIL:
                dummy_atoms = Atoms('OHH',positions = [[0,0,0],[-0.76,0.59,0],[0.76,0.59,0]])
                base_calculator = MbpolCalculator(dummy_atoms)
                calculators.append(base_calculator)
                continue
            else:
                raise Exception('module for MbpolCalculator not available')

        base_calculator = Siesta_Calculator(basis= settings_choice['basis' + i],
                                xc = settings_choice['xcfunctional' + i].upper())
        base_calculator.set_solution_method(settings_choice['solutionmethod' + i])

        if  model_path != None:
            if settings['ipp_client'] != None:
                client = ipp.Client(profile=settings['ipp_client'])
            else:
                client = None

            mlcf_calc = load_mlcf(model_path, client)
            mlcf_calc.cm_corrected = settings_choice['cmcorrection' + i]
            mlcf_calc.set_base_calculator(base_calculator)
            calculators.append(mlcf_calc)
        else:
            calculators.append(base_calculator)

    if settings['mixing']:
        calc = Mixer(calculators[0], calculators[1], mixing_settings['n'])
    else:
        calc = calculators[0]

    return calc

def load_mlcf(model_path, client = None):
    """ Given a directory model_path load and return the a calculator that uses
    the MLCF containes in that directory, for parallel computing an ipyparallel
    client can be provided
    """

    if not model_path[-1] == '/': model_path += '/'

    all_files = os.listdir(model_path)

    force_models = [f for f in all_files if 'force_' in f]
    species = [f[-1] for f in force_models]

    force_models = {s.lower(): load_force_model(model_path, s.lower())\
        for s in species}

    # Out of the element specific basis sets construct the full basis
    full_basis = {}
    for species in force_models:
        for entry in force_models[species].basis:
            if entry in full_basis and\
             full_basis[entry] != force_models[species].basis[entry]:
                raise Exception('Conflicting basis sets across force models')
            else:
                full_basis[entry] = force_models[species].basis[entry]


    descr_getter = DescriptorGetter(full_basis, client)

    masks = {}
    for s in force_models:
        masks[s.lower()] = force_models[s.lower()].mask

    scalers = {}
    for s in force_models:
        scalers[s.lower()] = force_models[s.lower()].scaler

    descr_getter.set_masks(masks)
    descr_getter.set_scalers(scalers)
    calc = MLCF_Calculator(feature_getter = descr_getter)
    calc.set_force_model(force_models)
    return calc
