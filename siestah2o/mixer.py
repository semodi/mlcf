import sys
import os
import shutil
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta

class Mixer(Siesta):

    def __init__(self, fast_calculator, slow_calculator, n):
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

    def get_potential_energy(self, atoms, force_consistent = False):

        f_fast = self.fast_calculator.get_forces(atoms)
        if self.step%self.n == 0:
            shutil.copy('H2O.DM','DM_save/DM.fast')
            try:
                shutil.copy('DM_save/DM.slow', 'H2O.DM')
            except FileNotFoundError:
                pass

            f_slow = self.slow_calculator.get_forces(atoms)
            self.forces = f_fast + self.n * (f_slow - f_fast)
            shutil.copy('H2O.DM','DM_save/DM.slow')
            shutil.copy('DM_save/DM.fast', 'H2O.DM')
            return self.slow_calculator.get_potential_energy(atoms)
        else:
            self.forces = f_fast
            return self.fast_calculator.get_potential_energy(atoms)

    def increment_step(self):
        self.step += 1

    def get_forces(self, atoms):
        self.get_potential_energy(atoms)
        return self.forces


