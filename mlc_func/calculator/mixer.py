import sys
import os
import shutil
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta
import numpy as np

class Mixer(Siesta):

    def __init__(self, fast_calculator, slow_calculator, n, correct_species = ''):
        try:
            shutil.os.mkdir('DM_save/')
        except FileExistsError:
            pass

        super().__init__(label = 'H2O')
        self.fast_calculator = fast_calculator
        self.slow_calculator = slow_calculator
        self.n = n
        self.step = 1
        self.energy = 0
        self.forces = 0
        self.correct_species = correct_species

    def get_potential_energy(self, atoms, force_consistent = False):
        calc_required = \
         self.fast_calculator.calculation_required(atoms, ['energy'])

        if calc_required:
            f_fast = self.fast_calculator.get_forces(atoms)
            if self.step%self.n == 0:
                shutil.copy('H2O.DM','DM_save/DM.fast')
                try:
                    shutil.copy('DM_save/DM.slow', 'H2O.DM')
                except FileNotFoundError:
                    pass

                f_slow = self.slow_calculator.get_forces(atoms)
                self.forces = f_fast + self.n * (f_slow - f_fast)
                if len(self.correct_species) > 0:
                    print('Only mixing ' + self.correct_species)
                    dont_correct = [s.lower() not in self.correct_species.lower() for s in atoms.get_chemical_symbols()]
                    self.forces[dont_correct] = f_fast[dont_correct]
                    try:
                        if self.fast_calculator.cm_corrected:
                            self.forces[dont_correct] -= np.mean(self.forces, axis = 0)    
                    except AttributeError:
                        pass

                with open('forces_mixing.dat', 'a') as file:
                    np.savetxt(file, f_slow - f_fast, fmt = '%.4f')

                shutil.copy('H2O.DM','DM_save/DM.slow')
                shutil.copy('DM_save/DM.fast', 'H2O.DM')
                self.energy = self.slow_calculator.get_potential_energy(atoms)
            else:
                self.forces = f_fast
                self.energy = self.fast_calculator.get_potential_energy(atoms)

            self.step += 1

        return self.energy

    def increment_step(self):
        self.step += 1

    def get_forces(self, atoms):
        self.get_potential_energy(atoms)
        return self.forces
