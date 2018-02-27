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
from siestah2o import SiestaH2O
import os

# Parse Command line
parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')

parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=1.0, help='Timestep in fs')
parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=10, help='Number of timesteps between MC')
parser.add_argument('-Nmax', metavar='Nmax', type=int, nargs = '?', default=1000, help='Max. number of MC steps')
parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./verlet_mc_results/', help='Save in directory')
parser.add_argument('-xc', metavar='xc', type=str, nargs = '?', default='BH', help='Which XC functional?')
parser.add_argument('-basis', metavar='basis', type=str, nargs= '?', default='qz', help='Basis: qz or dz')

args = parser.parse_args()

print('\n============ Starting calculation ========== \n \n')
print('Temperature = {} K'.format(args.T))
print('Timesteps = {} fs'.format(args.dt))
print('Number of steps per MC = {}'.format(args.Nt))
print('Max. number of MD steps = {}'.format(args.Nmax))
print('Save results in ' + args.dir)
print('Use functional ' + args.xc)
print('Use basis ' + args.basis)
print('\n===========================================\n')

try:
    shutil.os.mkdir(args.dir)
except FileExistsError:
    pass


file_extension = '{}_{}_{}_{}'.format(int(args.T),int(args.dt*1000),int(args.Nt),int(args.Nmax))

# Load initial configuration

a = 15.646

h2o = Atoms('128OHH',
            positions = np.genfromtxt('128.csv',delimiter = ','),
            cell = [a, a, a],
            pbc = True)
h2o_siesta = Atoms('128OHH',
            positions = np.genfromtxt('128.csv',delimiter = ','),
            cell = [a, a, a],
            pbc = True)

try:
    os.chdir('./siesta/')
except FileNotFoundError:
    os.mkdir('./siesta/')
    os.chdir('./siesta/')

mbp.reconnect_monomers(h2o)

h2o.calc = mbp.MbpolCalculator(h2o)
h2o_siesta.calc = SiestaH2O(basis = args.basis, xc = args.xc)

def shuffle_momenta(h2o):
    MaxwellBoltzmannDistribution(h2o, args.T * ase_units.kB)
    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))


shuffle_momenta(h2o)

dyn = VelocityVerlet(h2o, args.dt * ase_units.fs)

#prop = pimd.PIMDPropagator(h2o, steps = args.Nt, output_freq = 1)


positions = []

E0 = h2o.get_kinetic_energy() + h2o_siesta.get_potential_energy()
pos0 = h2o.get_positions()

rands = np.random.rand(args.Nmax)
temperature = args.T * ase_units.kB

trajectory = Trajectory(args.dir + 'verlet' + file_extension + '_keepv_dc.traj', 'a')
my_log = logger.MDLogger(dyn, h2o,args.dir + 'verlet' + file_extension + '_keepv_dc.log')

for i in range(args.Nmax):
    print('Propagating...')
    dyn.run(args.Nt)
    print('Propagating done. Calculating new energies...')
#   prop.set_atoms(h2o)
#   prop.propagate()
    h2o_siesta.set_positions(h2o.get_positions())
    E1 = h2o.get_kinetic_energy() + h2o_siesta.get_potential_energy()
    de = E1 - E0
    print('Energy difference: {} eV'.format(de))

    if de <= 0:
        pos1 = h2o.get_positions()
        positions.append(pos1)
        pos0 = np.array(pos1)
        trajectory.write(h2o)
        my_log()
#        shuffle_momenta(h2o)
        E0 = h2o.get_kinetic_energy() + h2o_siesta.get_potential_energy()
    else:
        if rands[i] < np.exp(-de/temperature):
            pos1 = h2o.get_positions()
            positions.append(pos1)
            pos0 = np.array(pos1)
            trajectory.write(h2o)
            my_log()
#            shuffle_momenta(h2o)
            E0 = h2o.get_kinetic_energy() + h2o_siesta.get_potential_energy()
        else:
            h2o.set_positions(pos0)
            h2o_siesta.set_positions(pos0)
            shuffle_momenta(h2o)
            E0 = h2o.get_kinetic_energy() + h2o_siesta.get_potential_energy()
            continue
