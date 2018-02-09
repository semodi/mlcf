from ase import Atoms
from ase.optimize import BFGS
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.calculators.nwchem import NWChem
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write
import sys
import numpy as np
from ase import units as ase_units
from ase.io import Trajectory
from ase.io import read
import pandas as pd
import time
import hashlib
import shutil
import os
import subprocess
eVtokcal = 23.06035
kcaltoeV = 1/eVtokcal

#TODO: also intialize velocities (or make it an option), read coordinates/veloctities and return them 
class PIMDPropagator:
      
    def __init__(self, atoms, steps=10,
             dt=0.5, output_freq=-1,
             thermostat='none', tau=0.01, 
             temperature=300):
        
        self.filename = '.' + hashlib.md5(str(time.time()).encode()).hexdigest()                 
        self.atoms = atoms
        self.steps = steps
        self.dt = dt
        if output_freq == -1:
            self.output_freq = steps
        else:
            self.output_freq = output_freq
        self.thermostat = thermostat  
        self.tau = tau
        self.temperature = temperature      

        self._update_input_file()         
        self._update_xyz_file()
        
    def __del__(self):
        os.remove(self.filename + '.inp')
        os.remove(self.filename + '.xyz')

    def _update_xyz_file(self):
        write(self.filename + '.xyz', self.atoms)
        with open(self.filename + '.xyz', 'r+') as file:
            n_atoms = file.readline()
            sidelength = file.readline().split()[4]
            file.seek(0)
            file.write('{} \t {} \t {} \n'.format(*([sidelength]*3)))
            file.write(n_atoms)
            
    def _update_input_file(self):
        with open('.pimd_input.template','r') as inputfile:
            input_template = inputfile.read()
            replace_dict = {'positions_file': "'" + self.filename + ".xyz'",
                            'name': "'" + self.filename + "'",
                            'num_steps': self.steps,
                            'time_step': self.dt,
                            'output_freq': self.output_freq,
                            'thermostat': self.thermostat,
                            'tau': self.tau,
                            'temperature': self.temperature}
            for key in replace_dict:
                input_template = input_template.replace('PAR_' + key,
                                                      str(replace_dict[key]))
        with open(self.filename + '.inp','w') as inputfile:
            inputfile.write(input_template)
        
    def calculation_required(self, atoms, quantities):
        return True
    
    def is_calculated(self, atoms):
        return (not np.all(self.last_coordinates == atoms.positions))
    
    def get_forces(self, atoms):
        pass 
    
    def get_potential_energy(self, atoms, force_consistent = False):
        pass 

    def set_atoms(self, atoms):
        self.atoms = atoms
        self._update_xyz_file()

    def get_atoms(self, atoms):
        return self.atoms 

    def propagate(self):
        cmd = ['PIMD.x ' + self.filename +'.inp']
        subprocess.check_call(cmd, shell = True)
                

    def get_stress(self, atoms):
        return np.zeros([3,3])
        raise Exception('Not implemented')

if __name__ == '__main__':
    h2o = read('128.xyz')
    h2o.set_cell([[15.646,0,0],[0,15.646,0],[0,0,15.646]])
    h2o.set_pbc(True)
    prop = PIMDPropagator(h2o)
    prop.propagate()
