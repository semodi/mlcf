""" This driver can be used to run MD simulations using Siesta together with MLCFs
"""
#!/usr/bin/env python
import mlc_func as mlcf
import os
from ase import Atoms
from ase.md.npt import NPT
from ase.md import VelocityVerlet, Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read,write, Trajectory
from ase import io
from ase import units as ase_units
from ase.md import logger
import numpy as np
import shutil
from mlc_func.timer import Timer
import argparse
try:
    from mbpol_calculator import MbpolCalculator
    MBPOL_AVAIL = True
except ImportError:
    MBPOL_AVAIL = False
from mlc_func.md import ListIntegrator

os.environ['SIESTA_COMMAND'] = 'siesta < ./%s > ./%s'

def log_all(energies = None, forces= None, coords = None):
    with open('energies.dat', 'a') as file:
        file.write('{:.4f}\n'.format(energies))
    with open('forces.dat', 'a') as file:
        np.savetxt(file, forces, fmt = '%.4f')
    with open('coords.dat', 'a') as file:
        np.savetxt(file, coords, fmt = '%.4f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a molecular dynamics simulation using MLCF calculators')
    parser.add_argument('startfile', action='store', help ='Path to .xyz/.traj file containing starting configuration')
    parser.add_argument('calcfile', action='store', help='Path to .inp file defining the calculator')
    parser.add_argument('name', action='store', help='Name of this run')
    parser.add_argument('-integrator', metavar='integrator', type=str, default='VV', nargs = '?', help ='Integrator: Choose VV (Velocity Verlet, default) or NH (Nose-Hoover)')
    parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
    parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=0.5, help='Timestep in fs')
    parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=1e6, help='Number of timesteps')
    parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./', help='Save in directory')
    parser.add_argument('-ttime', metavar='ttime', type=int, nargs= '?', default=10, help='ttime for NH thermostat')
    parser.add_argument('-restart', action=('store_true'), help='Restart last calculation')
    parser.add_argument('-np', metavar='np', type=int, nargs = '?', default=1, help='Number of mpi processes')
    parser.add_argument('-pseudoloc', metavar='pseudoloc', type=str, nargs = '?', default='./', help='Location of pseudopotentials')
    parser.add_argument('-log_every', metavar='log_every', type=int, nargs = '?', default=0,
        help='Log forces and energies every n steps')
    parser.add_argument('-cmcorrected', metavar='cmcorrected', type=int, nargs = '?', default=0,
        help='Set CM momentum = 0 every n steps')


    args = parser.parse_args()
    ttime = args.ttime
    if args.np > 1:
        os.environ['SIESTA_COMMAND'] = 'mpirun -np {} siesta < ./%s > ./%s'.format(args.np)
    else:
        os.environ['SIESTA_COMMAND'] = 'siesta < ./%s > ./%s'.format(args.np)

    os.environ['SIESTA_PP_PATH'] = args.pseudoloc
    args.Nt = int(args.Nt)
    # Load initial configuration
    if not args.dir[-1] =='/': args.dir += '/'
    if args.integrator.lower() == 'list':
        atoms = read(args.startfile, ':')
    elif args.restart:
        atoms = read(args.dir + args.name + '.traj')
    else:
        atoms = read(args.startfile)
        MaxwellBoltzmannDistribution(atoms, args.T * ase_units.kB)
        print('ttime= {} fs :: temperature = {}'.format(args.ttime, atoms.get_temperature()))
        atoms.set_momenta(atoms.get_momenta() - np.mean(atoms.get_momenta(), axis =0))

    calcdir = os.getcwd()
    args.calcfile = calcdir + '/' + args.calcfile
    try:
        shutil.os.mkdir(args.dir)
    except FileExistsError:
        pass
    os.chdir(args.dir)

    workdir = args.startfile.split('.')[0] if args.integrator.lower() == 'list' else 'siesta'
    workdir = calcdir + '/' + workdir + '/'

    os.environ['SIESTA_WORKING_DIR'] = workdir
    try:
        shutil.os.mkdir(workdir)
    except FileExistsError:
        pass
    os.chdir(workdir)


    calculator = mlcf.md.load_calculator_from_file(args.calcfile)
    if not args.integrator.lower() == 'list':
        traj = io.Trajectory('../' + args.name  + '.traj',
                         mode = 'a', atoms = atoms)

    if isinstance(calculator, mlcf.md.calculator.Mixer):
        if isinstance(calculator.fast_calculator, MbpolCalculator):
            calculator.fast_calculator = MbpolCalculator(atoms)
        if isinstance(calculator.slow_calculator, MbpolCalculator):
            calculator.slow_calculator = MbpolCalculator(atoms)
    elif isinstance(calculator, MbpolCalculator):
        calculator = MbpolCalculator(atoms)

    if not args.integrator == 'list':
        atoms.calc = calculator

    if args.integrator.lower() == 'nh':
        dyn = NPT(atoms, timestep = args.dt * ase_units.fs,
                  temperature =  args.T * ase_units.kB, externalstress = 0,
                  ttime = args.ttime * ase_units.fs, pfactor = None,
                             trajectory=traj,
                             logfile= '../'+ args.name + '.log')
    elif args.integrator.lower() == 'vv':
        dyn = VelocityVerlet(atoms, dt = args.dt * ase_units.fs,
                             trajectory=traj,
                             logfile='../'+ args.name + '.log')
    elif args.integrator.lower() == 'lv':
         dyn = Langevin(atoms, timestep = args.dt * ase_units.fs,
                  temperature =  args.T * ase_units.kB, friction = 1e-3,
                             trajectory=traj,
                             logfile= '../'+ args.name + '.log')
    elif args.integrator.lower() == 'list':
        dyn = ListIntegrator(atoms, args.calcfile)
        args.Nt = 1
    else:
        raise Exception('Unknown integrator')

    for step in range(args.Nt):
        time = Timer('Timer')

        if args.cmcorrected > 0:
            if step%args.cmcorrected == 0:
                momenta = atoms.get_momenta()
                atoms.set_momenta(momenta - np.mean(momenta, axis=0))
                momenta = atoms.get_momenta()
                cmfile = open('cmcorrection','a')
                np.savetxt(cmfile, np.mean(momenta,axis=0) , fmt = '%.4f')
                cmfile.close()

        if args.log_every > 0:
            if step%args.log_every == 0:
                log_all(atoms.get_potential_energy(), atoms.get_forces(), atoms.get_positions())


        dyn.run(1)
        time.stop()
