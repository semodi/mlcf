from siestah2o_new import SiestaH2O, write_atoms, read_atoms, Timer
import numpy as np
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
import os
import ipyparallel as ipp
import time
#TODO: Get rid of absolute paths
#os.environ['QT_QPA_PLATFORM']='offscreen'
#try:
#    jobid = os.environ['PBS_JOBID']
#    client = ipp.Client(profile='mpi'+str(jobid))
#    view = client.load_balanced_view()
#    print('Clients operating : {}'.format(len(client.ids)))
#    n_clients = len(client.ids)
#except OSError:
#    print('Warning: running without ipcluster')
#    n_clients = 0
#if n_clients == 0:
#    n_clients = 1

if __name__ == '__main__':

    # Parse Command line
    parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')

    parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
    parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=1.0, help='Timestep in fs')
    parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=10, help='Number of timesteps')
    parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./verlet_mc_results/', help='Save in directory')
    parser.add_argument('-xc', metavar='xc', type=str, nargs = '?', default='pbe', help='Which XC functional?') 
    parser.add_argument('-basis', metavar='basis', type=str, nargs= '?', default='dz', help='Basis: qz or dz')
    parser.add_argument('-npe', metavar='npe', type=int, nargs= '?', default=1, help='Nodes per engine')
    parser.add_argument('-ppn', metavar='ppn', type=int, nargs= '?', default=16, help='Processors per node')
    parser.add_argument('-ttime', metavar='ttime', type=int, nargs= '?', default=0.1, help='ttime for NH thermostat')

    args = parser.parse_args()
    args.xc = args.xc.upper()
    args.basis = args.basis.lower()

    print('\n============ Starting calculation ========== \n \n')
    print('Temperature = {} K'.format(args.T))
    print('Timesteps = {} fs'.format(args.dt))
    print('Number of steps = {}'.format(args.Nt))
    print('Save results in ' + args.dir)
    print('Use functional ' + args.xc)
    print('Use basis ' + args.basis)
    print('Processors per node: {}'.format(args.ppn))
    print('Nodes per engine: {}'.format(args.npe))
    print('\n===========================================\n')

    file_extension = '{}_{}_{}_{}'.format(int(args.T),int(args.dt*1000),int(args.Nt),int(args.Nmax))

    # Load initial configuration
    
#    a = 15.646

#    h2o = Atoms('128OHH',
#                positions = read('128.xyz').get_positions(),
#                cell = [a, a, a],
#                pbc = True)

    b = 13.0
    h2o = Atoms('64OHH',
                positions = read('64.xyz').get_positions(),
                cell = [b, b, b],
                pbc = True)
 

    try:
        shutil.os.mkdir(args.dir)
    except FileExistsError:
        pass

    os.chdir(args.dir)

    h2o.calc = SiestaH2O(basis = args.basis, xc = args.xc)

    temperature = args.T * ase_units.kB

    # Setting the initial T 100 K lower leads to faster convergence of initial oscillations
    MaxwellBoltzmannDistribution(h2o, (args.T - 100) * ase_units.kB)
    print('ttime= {} fs :: temperature = {}'.format(ttime,h2o.get_temperature()))

    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))
    traj = io.Trajectory('md_siesta_{}.traj'.format(int(ttime)),
                         mode = 'a', atoms = h2o)

    dyn = NPT(h2o, timestep = args.dt * ase_units.fs,
              temperature =  temperature, externalstress = 0,
              ttime = args.ttime * ase_units.fs, pfactor = None,
                         trajectory=traj,
                         logfile='./md_siesta.log'.format(int(ttime)))

    dyn.run(args.Nt)
