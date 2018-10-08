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
from mbpol_calculator import reconnect_monomers
eVtokcal = 23.06035
kcaltoeV = 1/eVtokcal


class PIMDPropagator:
      
    def __init__(self, atoms, steps=10,
             dt=0.5, output_freq=-1,
             thermostat='none', tau=0.01, 
             temperature=300):
        
        self.filename = '.' + hashlib.md5(str(time.time()).encode()).hexdigest()                 
#        self.filename = '.tempfilename'
        self.atoms = atoms
        self.atoms.set_positions(self.atoms.get_positions(True))
        self.steps = steps
        self.dt = dt
        if output_freq == -1:
            self.output_freq = steps
        else:
            self.output_freq = output_freq
        self.thermostat = thermostat  
        self.tau = tau
        self.temperature = temperature      

        if not isinstance(self.atoms.get_velocities(),np.ndarray):
            self.atoms.set_velocities(np.zeros_like(atoms.get_positions()))

       
        self._update_input_file()         
        self._update_xyz_file()
        
    def __del__(self):
        os.remove(self.filename + '.inp')
        os.remove(self.filename + '.img')

    def _update_xyz_file(self):
        with open(self.filename + '.img', 'w') as file:
            file.write("{:10d}{:10d}\n".format(self.atoms.get_number_of_atoms(), 1))
            file.write("{:12.6f}  {:12.6} {:12.6} {:12.6} \n".format(0.001,*np.diag(self.atoms.get_cell())))
            labels = ['O ','H ','H ']
            for i, [c, v] in enumerate(zip(self.atoms.get_positions(), self.atoms.get_velocities())):
                file.write("{} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(labels[i%3],*c,*(v*98.22695)))
                   
    def _update_input_file(self):
        with open('.pimd_input.template','r') as inputfile:
            input_template = inputfile.read()
            replace_dict = {'positions_file': "'" + self.filename + ".img'",
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
    def get_forces(self, atoms):
        raise Exception('Not implemented') 
    
    def get_potential_energy(self, atoms, force_consistent = False):
        raise Exception('Not implemented') 

    def set_atoms(self, atoms):
        self.atoms = atoms
        reconnect_monomers(self.atoms)
        self._update_xyz_file()

    def get_atoms(self, atoms):
        return self.atoms 

    def propagate(self):
        cmd = ['PIMD.x ' + self.filename +'.inp']
        subprocess.check_call(cmd, shell = True)
        new_atoms = read('out_' + self.filename + '_coord.xyz', index = -1)
        self.atoms.set_positions(new_atoms.get_positions())
        reconnect_monomers(self.atoms)     
        reconnect_monomers(self.atoms)     
        reconnect_monomers(self.atoms)     
        self.atoms.set_momenta(read('out_' + \
             self.filename + '_momenta.dat', format='xyz',
             index = -1).get_positions()/(98.22695))
        subprocess.check_call(['rm out_.*'], shell = True)                
        return self.atoms
    
    def get_stress(self, atoms):
        return np.zeros([3,3])
        raise Exception('Not implemented')

 
if __name__ == '__main__':
    h2o = read('128.xyz')
    h2o_old = Atoms(h2o)
    h2o.set_cell([[15.646,0,0],[0,15.646,0],[0,0,15.646]])
    h2o.set_pbc(True)
    MaxwellBoltzmannDistribution(h2o, 300 * ase_units.kB)
    print(h2o.get_temperature()) 
    prop = PIMDPropagator(h2o, steps = 10, output_freq = 1)
    prop.propagate()
    print(h2o.get_temperature())
    print(h2o.get_velocities())
