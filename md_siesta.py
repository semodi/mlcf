from siestah2o import SiestaH2O, write_atoms, read_atoms, Timer, DescriptorGetter
import numpy as np
from ase import Atoms
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read,write, Trajectory
from ase import io
from ase import units as ase_units
from ase.md import logger
import argparse
import shutil
import os
import ipyparallel as ipp
import time
#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'
try:
    jobid = os.environ['PBS_JOBID']
    client = ipp.Client(profile='mpi'+str(jobid))
    view = client.load_balanced_view()
    print('Clients operating : {}'.format(len(client.ids)))
    n_clients = len(client.ids)
except OSError:
    print('Warning: running without ipcluster')
    n_clients = 0
if n_clients == 0:
    n_clients = 1

if __name__ == '__main__':

    # Parse Command line
    parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')

    parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
    parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=1.0, help='Timestep in fs')
    parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=10, help='Number of timesteps')
    parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./md_siesta_results/', help='Save in directory')
    parser.add_argument('-xc', metavar='xc', type=str, nargs = '?', default='pbe', help='Which XC functional?') 
    parser.add_argument('-basis', metavar='basis', type=str, nargs= '?', default='dz', help='Basis: qz or dz')
    parser.add_argument('-ttime', metavar='ttime', type=int, nargs= '?', default=10, help='ttime for NH thermostat')
    parser.add_argument('-fd', metavar='fd', type=str, nargs= '?', default='n', help='Use finite difference model')
    parser.add_argument('-corrected', metavar='corrected', type=str, nargs= '?', default='y', help='Use ML correction')
    parser.add_argument('-restart', metavar='restart', type=str, nargs= '?', default='n', help='Restart recent calculation')

    args = parser.parse_args()
    args.xc = args.xc.upper()
    args.basis = args.basis.lower()
    ttime = args.ttime
    
    use_fd = (args.fd == 'y')
    corrected = (args.corrected == 'y')
    restart = (args.restart == 'y')

    print('\n============ Starting calculation ========== \n \n')
    print('Temperature = {} K'.format(args.T))
    print('Timesteps = {} fs'.format(args.dt))
    print('Number of steps = {}'.format(args.Nt))
    print('Save results in ' + args.dir)
    print('Use functional ' + args.xc)
    print('Use basis ' + args.basis)
    print('Use finite difference model: {}'.format(use_fd))
    print('Use ML correctino: {}'.format(corrected))
    if restart:
        print('Restart calculation from: ' + args.dir + 'md_siesta.traj')
    print('\n===========================================\n')


    # Load initial configuration
    
    a = 15.646

    h2o = Atoms('128OHH',
                positions = read('128.xyz').get_positions(),
                cell = [a, a, a],
                pbc = True)
    
    if restart:
        last_traj = read(args.dir + 'md_siesta.traj', index = -1)
        h2o.set_positions(last_traj.get_positions())
        h2o.set_momenta(last_traj.get_momenta())

    try:
        shutil.os.mkdir(args.dir)
    except FileExistsError:
        pass

    os.chdir(args.dir)

    try:
        shutil.os.mkdir('siesta/')
    except FileExistsError:
        pass

    os.chdir('siesta/')

    os.environ['SIESTA_COMMAND'] =\
         'mpirun -n {} siesta < ./%s > ./%s'.format(os.environ['PBS_NP'])

    h2o.calc = SiestaH2O(basis = args.basis, xc = args.xc, corrected = corrected, use_fd = use_fd)

    if n_clients > 1:
        descr_getter = DescriptorGetter(client)
    else:
        descr_getter = DescriptorGetter()

    h2o.calc.set_feature_getter(descr_getter)
    temperature = args.T * ase_units.kB

    # Setting the initial T 100 K lower leads to faster convergence of initial oscillations
    if not restart:
        MaxwellBoltzmannDistribution(h2o, args.T * ase_units.kB)
        print('ttime= {} fs :: temperature = {}'.format(ttime,h2o.get_temperature()))

        h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))

    traj = io.Trajectory('../md_siesta.traj'.format(int(ttime)),
                         mode = 'a', atoms = h2o)

    dyn = NPT(h2o, timestep = args.dt * ase_units.fs,
              temperature =  temperature, externalstress = 0,
              ttime = args.ttime * ase_units.fs, pfactor = None,
                         trajectory=traj,
                         logfile='../md_siesta.log'.format(int(ttime)))

    time_step = Timer('Timer') 
    for i in range(args.Nt):
        time_step.start_timer()
        dyn.run(1)
        time_step.stop() 
