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
import config
import pickle
from siestah2o import MullikenGetter

#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
os.environ['SIESTA_UTIL_FRACTIONAL'] = '/gpfs/home/smdick/siesta-4.0.1/Util/VCA/fractional'
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
    N_water = 96
    a = 15.646 * (N_water/128)**(1/3)

    h2o = Atoms('NaCl(OHH){}'.format(N_water),
                positions = np.genfromtxt('NaCl98H2O.csv', delimiter = ','),
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

    h2o.calc = SiestaH2O(basis = args.basis, xc = args.xc, log_accuracy = True)
    h2o.calc.set_solution_method(args.solutionmethod)
    if corrected:
        e_model_found = False
        f_model_found = False
        fd_model_found = False

        if args.features == 'descr':
            if n_clients > 1:
                descr_getter = DescriptorGetter(client)
            else:
                descr_getter = DescriptorGetter()
            h2o.calc.set_feature_getter(descr_getter)
            h2o.calc.set_symmetry(config.par['descr']['sym'])

        elif args.features == 'mull':
            mull_getter = MullikenGetter(128)
            try:
                mull_getter.n_o_orb = config.par['mull']['n_o_orb'][args.basis.lower()]
                mull_getter.n_h_orb = config.par['mull']['n_h_orb'][args.basis.lower()]
            except KeyError:
                print('Error: No Mulliken model available for this basis set')
            h2o.calc.set_feature_getter(mull_getter)

        try:
            h2o.calc.set_nn_path(config.model_basepath + config.par[args.features]['nn'][args.basis.lower()])
            e_model_found = True
        except KeyError:
            print('No energy model found, proceeding ...')

        if not use_fd:
            try:
                krr_o = pickle.load(open(config.model_basepath + config.par[args.features]['krr_o'][args.basis.lower()], 'rb'))
                krr_h = pickle.load(open(config.model_basepath + config.par[args.features]['krr_h'][args.basis.lower()], 'rb'))
                h2o.calc.set_force_model(krr_o, krr_h)
                f_model_found = True
            except KeyError:
                raise Exception('Error: no force model found. Aborting...')
        elif e_model_found:
            try:
                krr_o_dx = pickle.load(open(config.model_basepath + config.par[args.features]['krr_o_dx'][args.basis.lower()], 'rb'))
                krr_h_dx = pickle.load(open(config.model_basepath + config.par[args.features]['krr_h_dx'][args.basis.lower()], 'rb'))
                h2o.calc.set_fd_model(krr_o_dx, krr_h_dx)
                fd_model_found = True
            except KeyError:
                raise Exception('Error: no finite difference model found. Aborting...')
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
