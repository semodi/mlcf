from ase import Atoms
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write
import sys
import numpy as np
from ase import units as ase_units
from ase.io import Trajectory
from ase.io import write
from ase.io import read
import pandas as pd
import time
sys.path.append('/home/sebastian/Documents/Code/mbpol_calculator')
print(sys.path)
from mbpol_calculator import *
eVtokcal = 23.06035
kcaltoeV = 1/eVtokcal

if __name__ == '__main__':
    

    if len(sys.argv) == 2:
        ttime = float(sys.argv[1])
    else:
        ttime = 10.0

    
    a = 15.646 
    boxsize = [a,a,a] * unit.angstrom

    h2o = Atoms('128OHH',
                positions = np.genfromtxt('128.csv',delimiter = ','),
                cell = [a, a, a],
                pbc = True)


    h2o_shifted = reconnect_monomers(h2o)

    h2o.calc = MbpolCalculator(h2o)
    
    MaxwellBoltzmannDistribution(h2o, 300 * ase_units.kB)

    while(abs(h2o.get_temperature() - 300) > 1):
        MaxwellBoltzmannDistribution(h2o, 300 * ase_units.kB)
    print('ttime= {} fs :: temperature = {}'.format(ttime,h2o.get_temperature()))
 
    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))

#    dyn = NPT(h2o, timestep = 0.5 * ase_units.fs, 
#            temperature =  300 * ase_units.kB, externalstress = 0,
#                  ttime = ttime * ase_units.fs, pfactor = None,
#                         trajectory='/gpfs/scratch/smdick/mbpol/md_128_{}_0.5.traj'.format(int(ttime)),
#                         logfile='/gpfs/scratch/smdick/mbpol/md_128_{}_0.5.log'.format(int(ttime)))
    dyn = VelocityVerlet(h2o, dt = 0.5 * ase_units.fs, 
                         trajectory='/gpfs/scratch/smdick/mbpol/pureverlet_128_{}_0.5.traj'.format(int(ttime)),
                         logfile='/gpfs/scratch/smdick/mbpol/pureverlet_128_{}_0.5.log'.format(int(ttime)))


    starttime = time.time() 
    dyn.run(80000) 
    endtime = time.time()
#    traj = Trajectory('/gpfs/scratch/smdick/mbpol/md_128_{}_0.5.traj'.format(int(ttime)))
#    write('/gpfs/scratch/smdick/mbpol/md_128_{}_0.5.xyz'.format(int(ttime)), traj)
    pd.DataFrame([endtime-starttime]).to_csv('/gpfs/scratch/smdick/mbpol/pureverlet_128_{}_0.5.time'.format(int(ttime)),
        index = None, header = None)

