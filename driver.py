from siestah2o import SiestaH2O, Timer, DescriptorGetter, SiestaH2OAtomic
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
import json
try:
    from mbpol_calculator import *
except ImportError:
    print('Mbpol calculator not found')

from read_input import settings, mixing_settings

#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
# os.environ['SIESTA_PP_PATH'] = '/home/sebastian/Documents/Code/siesta-4.0.1/psf/'
#os.environ['SIESTA_PP_PATH'] = '/home/sebastian/Documents/Physics/Code/siesta-4.1-b3/psf/'

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

def log_all(energy = None,forces=None):
    if energy == None: #Initialize
        with open('energies.dat', 'w') as file:
            file.write('Siesta\n')
        with open('forces.dat', 'w') as file:
            pass
    else:
        with open('energies.dat', 'a') as file:
            file.write('{:.4f}\n'.format(energy))
        with open('forces.dat', 'a') as file:
            np.savetxt(file, forces, fmt = '%.4f')

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
        if settings_choice['modelkind' + i] in ['elf','none','nn']:
            calc = SiestaH2OAtomic(basis= settings_choice['basis' + i],
                                xc= settings_choice['xcfunctional' + i].upper(),
                                log_accuracy = True)
            
            if  settings_choice['modelkind' + i] != 'none':
                with open(model_path +'basis.json','r') as basisfile:
                    basis = json.loads(basisfile.readline()) 

                method = settings_choice['modelkind' + i]
                if n_clients > 1:
                    descr_getter = DescriptorGetter(method, basis, client)
                else:
                    descr_getter = DescriptorGetter(method, basis)

            # if settings_choice['modelkind' + i] in ['atomic']:
            #     descr_getter.single_thread = single_thread_descriptors_atomic
            # else:
            #     descr_getter.single_thread = single_thread_descriptors_molecular
            #

                scaler_o = pickle.load(open(model_path + 'scaler_O','rb'))
                scaler_h = pickle.load(open(model_path + 'scaler_H','rb'))
                try:
                    mask_o = np.genfromtxt(model_path + 'mask_O',dtype='str')
                    mask_h = np.genfromtxt(model_path + 'mask_H',dtype='str')
                    mask_o = [{'True': True, 'False': False}[m] for m in mask_o]
                    mask_h = [{'True': True, 'False': False}[m] for m in mask_h]
                    masks = {'o': mask_o, 'h': mask_h}
                    descr_getter.set_masks(masks)
                except FileNotFoundError:
                    pass 

                scalers = {'o': scaler_o, 'h': scaler_h}
                descr_getter.set_scalers(scalers)
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
    if settings['integrator'] == 'none':
        frame_list = read('../' + settings['xyzpath'], ':')
        log_all()
        for frame in frame_list:
            time_step.start_timer()
            h2o.set_positions(frame.get_positions())
            
            energy = h2o.get_potential_energy()
            forces = h2o.get_forces()
            log_all(energy,forces)
            time_step.stop()
    else:
        for i in range(settings['Nt']):
            time_step.start_timer()
            dyn.run(1)
            if settings['mixing']:
                h2o.calc.increment_step()
            time_step.stop()


