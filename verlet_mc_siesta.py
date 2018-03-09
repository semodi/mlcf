from siestah2o import SiestaH2O, write_atoms, read_atoms, Timer
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

def summarize_times(n_engines, step = 1):

    with open("TIMES", 'a') as outfile:
        outfile.write("============== Step = {} ============== \n".format(step))
        n_tabs = 0
        try:
            with open("TIME_STEP",'r') as infile:
                outfile.write(n_tabs * '\t' + 'STEP: ' + infile.read()) 
        except FileNotFoundError:
            pass

        n_tabs = 1
        try:
            with open("TIME_STEP_MAIN",'r') as infile:
                outfile.write(n_tabs * '\t' + 'STEP_MAIN: ' + infile.read()) 
        except FileNotFoundError:
            pass

        n_tabs = 2
        for id in range(n_engines):
            outfile.write(n_tabs * '\t' + '--- Engine: {} ---\n'.format(id))
            for root, dirs, files in os.walk('./{}'.format(id)):
                for file in files:
                    if 'TIME_' in file:
                        with open("./{}/".format(id) + file, 'r') as infile:
                            outfile.write(n_tabs * '\t' + file[5:] + ' ' + infile.read()) 
        n_tabs = 1
        try:
            with open("TIME_STEP_SUB",'r') as infile:
                outfile.write(n_tabs * '\t' + 'STEP_SUB: ' + infile.read()) 
        except:
            FileNotFoundError

def prop_and_eval(engine_id, args, shuffle_momenta = False, initialize = False):
    import numpy as np
    import sys
    sys.path.append('/gpfs/home/smdick/md_routines')
    from siestah2o import SiestaH2O, write_atoms, read_atoms, Timer
    import mbpol_calculator as mbp
    from ase import Atoms
    from ase.md import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units as ase_units
    import shutil
    import os
    from ase.io import Trajectory
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.chdir('/gpfs/home/smdick/md_routines/{}/{}/'.format(args.dir, engine_id))
    os.environ['SIESTA_COMMAND'] = \
     'mpirun -n {} -f machinefile siesta < ./%s > ./%s'.format(args.npe*args.ppn)
 
    h2o_siesta = read_atoms('tmp.traj', args.basis, args.xc)

    if shuffle_momenta:
        MaxwellBoltzmannDistribution(h2o_siesta, args.T * ase_units.kB)
        h2o_siesta.set_momenta(h2o_siesta.get_momenta() - \
         np.mean(h2o_siesta.get_momenta(),axis =0))
    if initialize:
        E0 = 0
    else:
        E0 = h2o_siesta.get_total_energy()

    h2o_mbpol = Atoms(h2o_siesta)
    h2o_mbpol.calc = mbp.MbpolCalculator(h2o_mbpol)
    mbp.reconnect_monomers(h2o_mbpol)
    mbp.reconnect_monomers(h2o_mbpol)
    dyn = VelocityVerlet(h2o_mbpol, args.dt * ase_units.fs)
    time_mbpol = Timer('TIME_MBPOL')
    dyn.run(args.Nt)
    with open('../mbpol.energies','a') as mbpolfile:
        mbpolfile.write('{}\t{}\n'.format(engine_id, h2o_mbpol.get_potential_energy()))
    time_mbpol.stop()
    h2o_siesta.set_positions(h2o_mbpol.get_positions())
    time_siesta = Timer('TIME_SIESTA_TOT')
    E1 = h2o_siesta.get_total_energy()
    time_siesta.stop()
    traj_writer = write_atoms(h2o_siesta, 'tmp.traj')
    os.chdir('../')
    return E1 - E0

if __name__ == '__main__':

    # Read nodefile 
    with open(os.environ['PBS_NODEFILE'],'r') as nodefile:
        hosts = nodefile.read().split()
    print(hosts)
    _, idx = np.unique(hosts, return_index=True)
    nodes = np.array(hosts)[np.sort(idx)]

    # Parse Command line
    parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')

    parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
    parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=1.0, help='Timestep in fs')
    parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=10, help='Number of timesteps between MC')
    parser.add_argument('-Nmax', metavar='Nmax', type=int, nargs = '?', default=1000, help='Max. number of MC steps')
    parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./verlet_mc_results/', help='Save in directory')
    parser.add_argument('-xc', metavar='xc', type=str, nargs = '?', default='pbe', help='Which XC functional?') 
    parser.add_argument('-basis', metavar='basis', type=str, nargs= '?', default='dz', help='Basis: qz or dz')
    parser.add_argument('-npe', metavar='npe', type=int, nargs= '?', default=1, help='Nodes per engine')
    parser.add_argument('-ppn', metavar='ppn', type=int, nargs= '?', default=16, help='Processors per node')
    args = parser.parse_args()
    args.xc = args.xc.upper()
    args.basis = args.basis.lower()

    print('\n============ Starting calculation ========== \n \n')
    print('Temperature = {} K'.format(args.T))
    print('Timesteps = {} fs'.format(args.dt))
    print('Number of steps per MC = {}'.format(args.Nt))
    print('Max. number of MD steps = {}'.format(args.Nmax))
    print('Save results in ' + args.dir)
    print('Use functional ' + args.xc)
    print('Use basis ' + args.basis)
    print('Processors per node: {}'.format(args.ppn))
    print('Nodes per engine: {}'.format(args.npe))
    print('\n===========================================\n')

    if args.npe * n_clients > len(nodes):
        raise Exception('(Nodes per engine)x(# engines) > (# available nodes)')

    file_extension = '{}_{}_{}_{}'.format(int(args.T),int(args.dt*1000),int(args.Nt),int(args.Nmax))

    # Load initial configuration
    
    a = 15.646

    h2o_siesta = Atoms('128OHH',
                positions = read('128.xyz').get_positions(),
                cell = [a, a, a],
                pbc = True)

 #   b = 13.0
 #   h2o_siesta = Atoms('64OHH',
 #               positions = read('64.xyz').get_positions(),
 #               cell = [b, b, b],
 #               pbc = True)
 #

    
    time_init = Timer('TIME_STEP')
    try:
        shutil.os.mkdir(args.dir)
    except FileExistsError:
        pass

    os.chdir(args.dir)

    for i in range(n_clients):
        try:
            shutil.os.mkdir('{}'.format(i))
        except FileExistsError:
            pass
        with open('{}/machinefile'.format(i), 'w') as machinefile:
            for rep in range(args.npe):
                for u in range(args.ppn):
                    machinefile.write('{}\n'.format(nodes[i*args.npe+rep]))
    
    os.environ['SIESTA_COMMAND'] = \
     'mpirun -n {} -f machinefile siesta < ./%s > ./%s'.format(args.npe*args.ppn)


    h2o_siesta.calc = SiestaH2O(basis = args.basis, xc = args.xc)

    write_atoms(h2o_siesta, './0/tmp.traj', False) 
    prop_and_eval(0, args, True, initialize = True)
    h2o_siesta = read_atoms('./0/tmp.traj', args.basis, args.xc)

    rands = np.random.rand(args.Nmax)
    temperature = args.T * ase_units.kB

    dyn = VelocityVerlet(h2o_siesta, args.dt * ase_units.fs)
    


    h2o_tracker = Atoms(h2o_siesta) 
    trajectory = Trajectory('verlet' + file_extension + '_keepv_dc.traj', 'a')
    log = logger.MDLogger(dyn, h2o_tracker,'verlet' + file_extension + '_keepv_dc.log')
#    log_all = logger.MDLogger(dyn, h2o_siesta,'verlet' + file_extension + '_keepv_dc.all.log')

    trajectory.write(h2o_tracker)
    log()
    shuffle_momenta = False

    time_init.stop()
    summarize_times(1,'INIT')

    time_main = Timer("TIME_MAIN")
    for i in range(args.Nmax):
        time_step = Timer("TIME_STEP")
        for u in range(n_clients):
            write_atoms(h2o_siesta, './{}/tmp.traj'.format(u))

        id_list = list(range(n_clients))
        args_list = [args] * n_clients
        time_step_main = Timer("TIME_STEP_MAIN")
        if shuffle_momenta:
            shuffle_momenta_list = [shuffle_momenta] * n_clients
        else:
            shuffle_momenta_list = [False] + [True] * (n_clients-1)

        if n_clients > 1:
            de = \
                view.map_sync(prop_and_eval,id_list, args_list,
                    shuffle_momenta_list)
        else:
            de = \
               list(map(prop_and_eval, id_list, args_list,
                    shuffle_momenta_list))
        time_step_main.stop()
        time_step_sub = Timer("TIME_STEP_SUB")
        de = np.array(de)
        where_accepted = np.where(rands[i] <= np.exp(-de/temperature))[0]
        if len(where_accepted) > 0:
            which = where_accepted[0]
        else:
            which = np.argmin(de)

        h2o_siesta_new = read_atoms('./{}/tmp.traj'.format(which), args.basis, args.xc)
        de = de[which]

        print('Energy difference: {} eV'.format(de))
        if de <= 0 or rands[i] < np.exp(-de/temperature):
            with open('accepted_ids','a') as idfile:
                idfile.write('{}\n'.format(which))
            h2o_siesta = h2o_siesta_new
            trajectory.write(h2o_siesta)
            h2o_tracker.set_positions(h2o_siesta.get_positions())
            h2o_tracker.set_momenta(h2o_siesta.get_momenta())
            h2o_tracker.set_calculator(h2o_siesta.get_calculator())
            log()
            shuffle_momenta = False
        else:
            with open('accepted_ids','a') as idfile:
                idfile.write('{}\n'.format(-1))
            shuffle_momenta = True

        time_step_sub.stop()
        time_step.stop()    
        summarize_times(n_clients, i)
    time_main.stop()
