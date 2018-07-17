from siestah2o import SiestaH2O, Timer, DescriptorGetter, SiestaH2OAtomic
from siestah2o import single_thread_descriptors_atomic, single_thread_descriptors_molecular
from siestah2o import MullikenGetter, Mixer
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
import keras
try:
    from mbpol_calculator import *
except ImportError:
    print('Mbpol calculator not found')

from read_input import settings, mixing_settings

#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'
#os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
os.environ['SIESTA_PP_PATH'] = '/home/sebastian/Documents/Code/siesta-4.0.1/psf/'

#Try to run siesta with mpi if that fails run serial version
try:
    os.environ['SIESTA_COMMAND'] =\
             'mpirun -n {} siesta < ./%s > ./%s'.format(os.environ['PBS_NP'])
except KeyError:
        os.environ['SIESTA_COMMAND'] =\
             'siesta < ./%s > ./%s'

# Load ipyparallel engines for feature extraction, if fails do serially
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

    # Load initial configuration
    if settings['restart']:
        h2o = read(settings['name'] + '.traj')
    else:
        h2o = read(settings['xyzpath'])

    try:
        shutil.os.mkdir('siesta/')
    except FileExistsError:
        pass
    os.chdir('siesta/')

    for key in settings:
        print( '({}, {})'.format(key, settings[key]))
    if settings['mixing']:
        for key in mixing_settings:
            print( '({}, {})'.format(key, mixing_settings[key]))

    if settings['mixing']:
        iterate_over = ['1','2']
        settings_choice = mixing_settings
    else:
        iterate_over = ['']
        settings_choice = settings

    calculators = []

    for i in iterate_over:

        model_path = settings_choice['model' + i]

        #============RSD=====================
        if settings_choice['modelkind' + i] in ['atomic','none','molecular']:
            calc = SiestaH2OAtomic(basis= settings_choice['basis' + i],
                                xc= settings_choice['xcfunctional' + i].upper(),
                                log_accuracy = True)
            if n_clients > 1:
                descr_getter = DescriptorGetter(client)
            else:
                descr_getter = DescriptorGetter()

            if settings_choice['modelkind' + i] in ['atomic']:
                descr_getter.single_thread = single_thread_descriptors_atomic
            else:
                descr_getter.single_thread = single_thread_descriptors_molecular

            if  settings_choice['modelkind' + i] != 'none':

                scaler_o = pickle.load(open(model_path + 'scaler_O','rb'))
                scaler_h = pickle.load(open(model_path + 'scaler_H','rb'))
                descr_getter.set_scalers(scaler_o, scaler_h)
                calc.set_feature_getter(descr_getter)

                krr_o = keras.models.load_model(model_path + 'force_O')
                krr_h = keras.models.load_model(model_path + 'force_H')
                calc.set_force_model(krr_o, krr_h)

            calc.set_solution_method(settings_choice['solutionmethod' + i])
        #==============DM=====================
        elif settings_choice['modelkind' + i] == 'mulliken':
            calc = SiestaH2O(basis= settings_choice['basis' + i],
                                xc= settings_choice['xcfunctional' + i].upper(),
                                log_accuracy = True)

            mull_getter = MullikenGetter(int(len(h2o.get_positions()/3)))
            try:
                mull_getter.n_o_orb = config.par['mull']['n_o_orb'][settings_choice['basis' + i]]
                mull_getter.n_h_orb = config.par['mull']['n_h_orb'][settings_choice['basis' + i]]
            except KeyError:
                print('Error: No Mulliken model available for this basis set')

            calc.set_feature_getter(mull_getter)
            calc.set_nn_path(model_path + 'energy')
            krr_o = pickle.load(open(model_path + 'force_O', 'rb'))
            krr_h = pickle.load(open(model_path + 'force_H', 'rb'))
            calc.set_force_model(krr_o, krr_h)
            calc.set_solution_method(settings_choice['solutionmethod' + i])
        
        #============MB-pol====================
        elif settings_choice['modelkind' + i] =='mbpol':
            h2o = reconnect_monomers(h2o)
            h2o = reconnect_monomers(h2o)
            calc =  MbpolCalculator(h2o)

        calculators.append(calc)

    if settings['mixing']:
        h2o.calc = Mixer(calculators[0], calculators[1], mixing_settings['n'])
    else:
        h2o.calc = calculators[0]

    # Setting the initial T 100 K lower leads to faster convergence of initial oscillations
    if not settings['restart']:
        MaxwellBoltzmannDistribution(h2o, settings['t'] * ase_units.kB)
        print('ttime= {} fs :: temperature = {}'.format(settings['ttime'], h2o.get_temperature()))

    h2o.set_momenta(h2o.get_momenta() - np.mean(h2o.get_momenta(),axis =0))

    traj = io.Trajectory('../' + settings['name']  + '.traj',
                         mode = 'a', atoms = h2o)

    if settings['integrator'] == 'nh':
        dyn = NPT(h2o, timestep = settings['dt'] * ase_units.fs,
                  temperature =  settings['t'] * ase_units.kB, externalstress = 0,
                  ttime = settings['ttime'] * ase_units.fs, pfactor = None,
                             trajectory=traj,
                             logfile='../'+ settings['name'] + '.log')
    else:
        dyn = VelocityVerlet(h2o, dt = settings['dt'] * ase_units.fs,
                             trajectory=traj,
                             logfile='../'+ settings['name'] + '.log')

    time_step = Timer('Timer')
    for i in range(settings['Nt']):
        time_step.start_timer()
        dyn.run(1)
        if settings['mixing']:
            h2o.calc.increment_step()
        time_step.stop()
