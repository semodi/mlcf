import numpy as np
import sys
sys.path.append('~/Documents/Physics/Code/mbpol_calculator')
import mbpol_calculator as mbp
from ase import Atoms
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read,write, Trajectory
from ase import units as ase_units
from ase.md import logger
import argparse
import shutil
import pimd_wrapper as pimd

# Parse Command line

parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')
parser.add_argument('-T', metavar='T', type=float, nargs = '?', default= 298.15, help ='Temperature in Kelvin')
parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default= 1.0, help='Timestep in fs')
parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=5, help='Number of timesteps between MC')
parser.add_argument('-Nmax', metavar='Nmax', type=int, nargs = '?', default=100000, help='Max. number of MC steps')
parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./verlet_mc_results/', help='Save in directory')
parser.add_argument('-restart', metavar='restart', type=str, nargs = '?', default='n', help='Continue calculation where left of? (y/n)')
parser.add_argument('-noise', metavar='noise', type=float, nargs = '?', default= 0.0, help='Noise on energy prediction')

args = parser.parse_args()

print('\n============ Starting calculation ========== \n \n')
print('Temperature = {} K'.format(args.T))
print('Timesteps = {} fs'.format(args.dt))
print('Number of steps per MC = {}'.format(args.Nt))
print('Max. number of MD steps = {}'.format(args.Nmax))
print('Save results in ' + args.dir)
print('Restart? {}'.format(args.restart))
print('Noise: {}'.format(args.noise))
print('\n===========================================\n')

try:
    shutil.os.mkdir(args.dir)
except FileExistsError:
    pass

if args.restart =='y':
    args.restart = True
elif args.restart == 'n':
    args.restart = False
else:
    raise Exception('Restart argument not understood, choose "y" or "n"')

if args.noise > 0.0:
    file_extension = '{}_{}_{}_{}_{}_random'.format(int(args.T), int(args.dt*1000), int(args.Nt), int(args.Nmax), int(args.noise*1000))
else:
    file_extension = '{}_{}_{}_{}_random'.format(int(args.T), int(args.dt*1000), int(args.Nt), int(args.Nmax))

# Load initial configuration 

# Box size for right density when N_mol = 128 
a = 15.646 

h2o = Atoms('128OHH',
            positions = np.genfromtxt('128.csv',delimiter = ','),
            cell = [a, a, a],
            pbc = True)

# Restart an old calculation and append to files?
if args.restart:
    h2o.set_positions(read(args.dir + 'verlet_mc' + file_extension + '.traj',index = -1).get_positions())
    print('restarting old simulation')
else:
    print('new simulation')

h2o.set_positions(read(args.dir + 'cp_analyze/verlet_mc' + file_extension[:-7] + '.traj',index = -1).get_positions())
#########
#h2o.set_positions(read('/gpfs/scratch/smdick/mbpol/bckp/verlet_mc298_1000_5_100000.traj',index = -1).get_positions())
#########

mbp.reconnect_monomers(h2o)
h2o.calc = mbp.MbpolCalculator(h2o, noise = args.noise)

def shuffle_momenta(h2o):
    MaxwellBoltzmannDistribution(h2o, args.T * ase_units.kB)
    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))


shuffle_momenta(h2o)

dyn = VelocityVerlet(h2o, args.dt * ase_units.fs)

#prop = pimd.PIMDPropagator(h2o, steps = args.Nt, output_freq = 1)


positions = []

rands = np.random.rand(args.Nmax)
temperature = args.T * ase_units.kB

if args.restart:
    trajectory = Trajectory(args.dir + 'verlet_mc' + file_extension + '.traj', 'a')
else:
    trajectory = Trajectory(args.dir + 'verlet_mc' + file_extension + '.traj', 'w')

my_log = logger.MDLogger(dyn, h2o,args.dir + 'verlet_mc' + file_extension + '.log')

#my_log_all = logger.MDLogger(dyn, h2o,args.dir + 'verlet' + file_extension + '_all.log')
#trajectory_all = Trajectory(args.dir + 'verlet' + file_extension + '_all.traj', 'a')

#steps_file = open(args.dir + 'verlet' + file_extension + '.steps', 'w')

#for i in range(5):
#    prop.propagate()
#    shuffle_momenta(h2o)

E0 = h2o.get_total_energy()
print("Initial Energy = {}".format(E0))
pos0 = h2o.get_positions()
my_log()
trajectory.write(h2o)

for i in range(args.Nmax):
#    steps_file.write('\n')    
    dyn.run(args.Nt)
#    prop.set_atoms(h2o) 
#    prop.propagate()
    E1 = h2o.get_total_energy()
#    my_log_all()
#    trajectory_all.write(h2o)
    de = E1 - E0
#    print("E1 = {} :: E0 = {} :: de = {}".format(E1,E0,de) )
#    steps_file.write("E1 = {} :: E0 = {} :: de = {}".format(E1,E0,de))
    rand_n = np.random.randn(1)
    if rand_n > -0.3:
        pos1 = h2o.get_positions()
        positions.append(pos1)
        pos0 = np.array(pos1)
        trajectory.write(h2o)
        my_log()
#        shuffle_momenta(h2o)
        E0 = h2o.get_total_energy()
#        steps_file.write("  :: accepted")
        print('accepted')
    else:
        h2o.set_positions(pos0)
        shuffle_momenta(h2o)
        E0 = h2o.get_total_energy() 
#            steps_file.write("  :: rejected")
        print('rejected')
        continue
#steps_file.close()
