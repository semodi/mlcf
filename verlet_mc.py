import numpy as np
import sys
sys.path.append('../')
import mbpol_wrapper as mbp
from ase import Atoms
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read,write, Trajectory
from ase import units as ase_units
from ase.md import logger
import argparse
import shutil


# Parse Command line

parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')

parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=1.0, help='Timestep in fs')
parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=10, help='Number of timesteps between MC')
parser.add_argument('-Nmax', metavar='Nmax', type=int, nargs = '?', default=1000, help='Max. number of MC steps')
parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./verlet_mc_results/', help='Save in directory')


args = parser.parse_args()

print('\n============ Starting calculation ========== \n \n')
print('Temperature = {} K'.format(args.T))
print('Timesteps = {} fs'.format(args.dt))
print('Number of steps per MC = {}'.format(args.Nt))
print('Max. number of MD steps = {}'.format(args.Nmax))
print('Save results in ' + args.dir)
print('\n===========================================\n')

try:
    shutil.os.mkdir(args.dir)
except FileExistsError:
    pass


file_extension = '{}_{}_{}_{}'.format(int(args.T),int(args.dt*1000),int(args.Nt),int(args.Nmax))

# Load initial configuration 

a = 15.646 
boxsize = [a,a,a] * unit.angstrom

h2o = Atoms('128OHH',
            positions = np.genfromtxt('128.csv',delimiter = ','),
            cell = [a, a, a],
            pbc = True)

h2o_shifted = mbp.reconnect_monomers(h2o)

h2o.calc = mbp.MbpolCalculator(h2o)

def shuffle_momenta(h2o):
    MaxwellBoltzmannDistribution(h2o, args.T * ase_units.kB)
    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))


shuffle_momenta(h2o)

dyn = VelocityVerlet(h2o, args.dt * ase_units.fs)


positions = []

E0 = h2o.get_total_energy()
pos0 = h2o.get_positions()

rands = np.random.rand(args.Nmax)
temperature = args.T * ase_units.kB

trajectory = Trajectory(args.dir + 'verlet' + file_extension + '_keepv_dc.traj', 'a')
my_log = logger.MDLogger(dyn, h2o,args.dir + 'verlet' + file_extension + '_keepv_dc.log')

for i in range(args.Nmax):
    
    dyn.run(args.Nt) 
    E1 = h2o.get_total_energy()
    de = E1 - E0
    if de <= 0:
        pos1 = h2o.get_positions()
        positions.append(pos1)
        pos0 = np.array(pos1)
        trajectory.write(h2o)
        my_log()
#        shuffle_momenta(h2o)
        E0 = h2o.get_total_energy()
    else:
        if rands[i] < np.exp(-de/temperature):
            pos1 = h2o.get_positions()
            positions.append(pos1)
            pos0 = np.array(pos1)
            trajectory.write(h2o)
            my_log()
#            shuffle_momenta(h2o)
            
            E0 = h2o.get_total_energy() 
        else:
            h2o.set_positions(pos0)
            shuffle_momenta(h2o)
            E0 = h2o.get_total_energy() 
            continue
