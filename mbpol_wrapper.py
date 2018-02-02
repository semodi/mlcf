from ase import Atoms
from ase.optimize import BFGS
from ase.md.npt import NPT
from ase.calculators.nwchem import NWChem
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
import sys
import mbpol
import numpy as np
from ase import units as ase_units
from ase.io import Trajectory
from ase.io import write
from ase.io import read

eVtokcal = 23.06035
kcaltoeV = 1/eVtokcal

kilocalorie_per_mole_per_angstrom = unit.kilocalorie_per_mole/unit.angstrom

class MbpolCalculator:
      
    def __init__(self, pdb, nonbondedMethod, nonbondedCutoff, boxSize = None, tip4p = False):
        
        self.tip4p = tip4p
        if tip4p:
            raise Exception( 'TIP4PFB not working at the moment')
            self.forcefield = app.ForceField("tip4pfb.xml")
        else:
            self.forcefield = app.ForceField(mbpol.__file__.replace("mbpol.py", "mbpol.xml"))
        
        self.nonbondedMethod = nonbondedMethod
        
        if self.nonbondedMethod == app.PME:
            if boxSize == None:
                raise Exception('specify box size')
            else:
                pdb.topology.setUnitCellDimensions(boxSize)
            
        self.system = self.forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=nonbondedMethod,
                                         nonbondedCutoff=nonbondedCutoff)
        
        self.platform = mm.Platform.getPlatformByName('Reference')
        integrator = mm.VerletIntegrator(0.02*unit.femtoseconds)
        self.simulation = app.Simulation(pdb.topology, self.system, integrator, self.platform)
        
        
        
        self.simulation.context.setPositions(pdb.positions) #ASE: Angstrom , OMM : nm
        self.simulation.context.computeVirtualSites()
        self.state = self.simulation.context.getState(getForces=True, getEnergy=True) 
        self.last_coordinates = np.array(pdb.positions.value_in_unit(unit.angstrom))
        
        self.last_coordinates = np.delete(self.last_coordinates,
                                          np.arange(3,len(self.last_coordinates),4), axis = 0).reshape(-1,3)
        
    def calculation_required(self, atoms, quantities):
        return True
    
    def is_calculated(self, atoms):
        return (not np.all(self.last_coordinates == atoms.positions))
    
    def get_forces(self, atoms):
        if self.is_calculated(atoms):
            pos = np.zeros([int(len(atoms.positions)/3*4),3]).reshape(-1,4,3)
            pos[:,:3,:] = atoms.positions.reshape(-1,3,3)
            self.simulation.context.setPositions(pos.reshape(-1,3)/10) #ASE: Angstrom , OMM : nm
            self.simulation.context.computeVirtualSites()
            self.state = self.simulation.context.getState(getForces=True, getEnergy=True)
            self.last_coordinates = np.array(atoms.positions)
        forces = np.array(self.state.getForces().value_in_unit(kilocalorie_per_mole_per_angstrom))*kcaltoeV
        forces = np.delete(forces, np.arange(3,len(forces),4), axis = 0)
        if self.tip4p:
            forces = np.tile(forces.reshape(-1,3,3).sum(axis=1).reshape(-1,1,3), [1,3,1]).reshape(-1,3)
        return forces
       
    
    def get_potential_energy(self, atoms, force_consistent = False):
        if self.is_calculated(atoms):
            pos = np.zeros([int(len(atoms.positions)/3*4),3]).reshape(-1,4,3)
            pos[:,:3,:] = atoms.positions.reshape(-1,3,3)
            self.simulation.context.setPositions(pos.reshape(-1,3)/10) #ASE: Angstrom , OMM : nm
            self.simulation.context.computeVirtualSites()
            self.state = self.simulation.context.getState(getEnergy=True, getForces=True)
            self.last_coordinates = np.array(atoms.positions)
        return self.state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)*kcaltoeV
        
    def get_stress(self, atoms):
        return np.zeros([3,3])
        raise Exception('Not implemented')

def reconnect_monomers(atoms, boxsize):
    pos0 = np.array(atoms.positions)
    
    for i,_ in enumerate(atoms.get_positions()[::3]):
        
        if atoms.get_distance(i*3,i*3+1) > min(boxsize) - 5:
            d = h2o.positions[i*3] - atoms.positions[i*3+1]
            which = np.where(np.abs(d) > 5)[0]
            for w in which:
                pos0[i*3+1, w] += d[w]/np.abs(d[w]) * boxsize[w]
        elif atoms.get_distance(i*3,i*3+2) > min(boxsize) -5:
            d = atoms.positions[i*3] - atoms.positions[i*3+2]
            which = np.where(np.abs(d) > 5)[0]
            for w in which:
                pos0[i*3+2, w] += d[w]/np.abs(d[w]) * boxsize[w]
            
    atoms.set_positions(pos0)
    
    return atoms


if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        ttime = sys.argv[1]
    else:
        ttime = 25

    pdb = app.PDBFile("../ase/128.pdb")
    init_pos = np.array(pdb.positions.value_in_unit(unit.angstrom))
    init_pos = np.delete(init_pos, np.arange(3,len(init_pos),4), axis = 0)

    a = 15.646 
    boxsize = [a,a,a] * unit.angstrom

    h2o = Atoms('128OHH',
                positions = init_pos,
                cell = [a, a, a],
                pbc = True)


    h2o_shifted = reconnect_monomers(h2o,[a,a,a])

    h2o.calc = MbpolCalculator(pdb, app.PME, boxSize = boxsize, nonbondedCutoff= 0.7*unit.nanometer, tip4p = False)

    MaxwellBoltzmannDistribution(h2o, 300 * ase_units.kB)

    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))

    dyn = NPT(h2o, timestep = 1.0 * ase_units.fs, temperature =  300 * ase_units.kB, externalstress = 0,
                  ttime = ttime * ase_units.fs, pfactor = None,
                         trajectory='md_128_{}.traj'.format(ttime), logfile='md_128_{}.log'.format(ttime))
    dyn.run(500) 

    traj = Trajectory('./md_128_{}.traj'.format(ttime))
    write('md_128.xyz', traj)
