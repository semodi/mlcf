from siestah2o import SiestaH2OAtomic, write_atoms, read_atoms, Timer, DescriptorGetter, AtomicGetter
import keras
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
import config
import pickle
from siestah2o import MullikenGetter

#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
#os.environ['SIESTA_PP_PATH'] = '/home/sebastian/Documents/Code/siesta-4.0.1/psf/'

try:
    os.environ['SIESTA_COMMAND'] =\
             'mpirun -n {} siesta < ./%s > ./%s'.format(os.environ['PBS_NP'])
except KeyError:
        os.environ['SIESTA_COMMAND'] =\
             'siesta < ./%s > ./%s'

try:
    jobid = os.environ['PBS_JOBID']
    client = ipp.Client(profile='mpi'+str(jobid))
    view = client.load_balanced_view()
    print('Clients operating : {}'.format(len(client.ids)))
    n_clients = len(client.ids)
except KeyError:
    print('Warning: running without ipcluster')
    n_clients = 0
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
    parser.add_argument('-basis', metavar='basis', type=str, nargs= '?', default='dz_custom', help='Basis: qz or dz')
    parser.add_argument('-ttime', metavar='ttime', type=int, nargs= '?', default=10, help='ttime for NH thermostat')
    parser.add_argument('-fd', metavar='fd', type=str, nargs= '?', default='n', help='Use finite difference model')
    parser.add_argument('-corrected', metavar='corrected', type=str, nargs= '?', default='y', help='Use ML correction')
    parser.add_argument('-restart', metavar='restart', type=str, nargs= '?', default='n', help='Restart recent calculation')
    parser.add_argument('-features', metavar='features', type=str, nargs='?', default='descr', help='descr/mull')
    parser.add_argument('-solutionmethod', metavar='solutionmethod', type=str, nargs='?', default='diagon', help='diagon/OMM')

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
    print('Use ML correction: {}'.format(corrected))
    if restart:
        print('Restart calculation from: ' + args.dir + 'md_siesta.traj')
    print('As features use: ' + args.features)
    print('Solution Method: ' + args.solutionmethod)
    if args.solutionmethod == 'OMM':
        print('Proceed with caution!')
    print('\n===========================================\n')


    # Load initial configuration


    h2o = read('start.traj')
#    h2o = read('mono.traj')

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

    h2o.calc = SiestaH2OAtomic(basis = args.basis, xc = args.xc, log_accuracy = True)
    h2o.calc.set_solution_method(args.solutionmethod)
    if corrected:
        e_model_found = False
        f_model_found = False
        fd_model_found = False

        if args.features == 'descr':
            if n_clients > 1:
                descr_getter = AtomicGetter(client)
            else:
                descr_getter = AtomicGetter()
            h2o.calc.set_feature_getter(descr_getter)

        if not use_fd:
            try:
                krr_o = keras.models.load_model(config.model_basepath + 'atom_force_O_descr')
                krr_h = keras.models.load_model(config.model_basepath + 'atom_force_H_descr')
                h2o.calc.set_force_model(krr_o, krr_h)
                f_model_found = True
            except KeyError:
                raise Exception('Error: no force model found. Aborting...')
        else:
            raise Exception('Error: finite difference model cannot be used as energy model not loaded')

    temperature = args.T * ase_units.kB

    # Setting the initial T 100 K lower leads to faster convergence of initial oscillations
    if not restart:
        MaxwellBoltzmannDistribution(h2o, args.T * ase_units.kB)
        print('ttime= {} fs :: temperature = {}'.format(ttime,h2o.get_temperature()))

    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))

    traj = io.Trajectory('../md_siesta.traj'.format(int(ttime)),
                         mode = 'a', atoms = h2o)

    dyn = VelocityVerlet(h2o, dt = args.dt * ase_units.fs,
                         trajectory=traj,
                         logfile='../md_siesta.log'.format(int(ttime)))

    time_step = Timer('Timer')
    for i in range(args.Nt):
        time_step.start_timer()
        dyn.run(1)
#        h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))
        time_step.stop()
