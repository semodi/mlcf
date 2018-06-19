from ase import Atoms
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.io as io
import numpy as np
from ase import units as ase_units
from mbpol_calculator import *

T = 298.15
dt = 2.5

if __name__ == '__main__':
    

    if len(sys.argv) == 2:
        ttime = float(sys.argv[1])
    else:
        ttime = 10.0
   
    a = 15.646 

    h2o = Atoms('128OHH',
                positions = np.genfromtxt('128.csv',delimiter = ','),
                cell = [a, a, a],
                pbc = True)
#    h2o = io.read('/gpfs/scratch/smdick/mbpol/nose_128_{}.traj'.format(int(ttime)), -1)

    h2o = reconnect_monomers(h2o)
    h2o = reconnect_monomers(h2o)

    h2o.calc = MbpolCalculator(h2o)
    
    # Setting the initial T 100 K lower leads to faster convergence of initial oscillations
    MaxwellBoltzmannDistribution(h2o, T * ase_units.kB)

#    while(abs(h2o.get_temperature() - 300) > 1):
#        MaxwellBoltzmannDistribution(h2o, 300 * ase_units.kB)
    print('ttime= {} fs :: temperature = {}'.format(ttime,h2o.get_temperature()))
 
    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))
    traj = io.Trajectory('/gpfs/scratch/smdick/mbpol/nose_128_{}_longstep.traj'.format(int(ttime)),
                         mode = 'a', atoms = h2o) 

    dyn = NPT(h2o, timestep = dt * ase_units.fs, 
              temperature =  T * ase_units.kB, externalstress = 0,
              ttime = ttime * ase_units.fs, pfactor = None,
                         trajectory=traj,
                         logfile='/gpfs/scratch/smdick/mbpol/nose_128_{}_longstep.log'.format(int(ttime)))

#    dyn = VelocityVerlet(h2o, dt = 0.5 * ase_units.fs, 
#                         trajectory='/gpfs/scratch/smdick/mbpol/pureverlet_128_{}_0.5.traj'.format(int(ttime)),
#                         logfile='/gpfs/scratch/smdick/mbpol/pureverlet_128_{}_0.5.log'.format(int(ttime)))
    dyn.run(100000) 

