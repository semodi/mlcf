from siestah2o import SiestaH2O
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

# Parse Command line
parser = argparse.ArgumentParser(description='Do hybrid MC with Verlet')

parser.add_argument('-T', metavar='T', type=float, nargs = '?', default=300.0, help ='Temperature in Kelvin')
parser.add_argument('-dt', metavar='dt', type=float, nargs = '?', default=1.0, help='Timestep in fs')
parser.add_argument('-Nt', metavar='Nt', type=int, nargs = '?', default=10, help='Number of timesteps between MC')
parser.add_argument('-Nmax', metavar='Nmax', type=int, nargs = '?', default=1000, help='Max. number of MC steps')
parser.add_argument('-dir', metavar='dir', type=str, nargs = '?', default='./verlet_mc_results/', help='Save in directory')
parser.add_argument('-xc', metavar='xc', type=str, nargs = '?', default='pbe', help='Which XC functional?')
parser.add_argument('-basis', metavar='basis', type=str, nargs= '?', default='dz', help='Basis: qz or dz')

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
print('\n===========================================\n')

file_extension = '{}_{}_{}_{}'.format(int(args.T),int(args.dt*1000),int(args.Nt),int(args.Nmax))

# Load initial configuration

a = 15.646
b = 13.0
#h2o = Atoms('128OHH',
#            positions = np.genfromtxt('128.csv',delimiter = ','),
#            cell = [a, a, a],
#            pbc = True)
#h2o_siesta = Atoms('128OHH',
#            positions = np.genfromtxt('128.csv',delimiter = ','),
#            cell = [a, a, a],
#            pbc = True)

h2o_mbpol = Atoms('64OHH',
            positions = read('64.xyz').get_positions(),
            cell = [b, b, b],
            pbc = True)
h2o_siesta = Atoms('64OHH',
            positions = read('64.xyz').get_positions(),
            cell = [b, b, b],
            pbc = True)

try:
    os.chdir('./siesta/')
except FileNotFoundError:
    os.mkdir('./siesta/')
    os.chdir('./siesta/')

try:
    shutil.os.mkdir(args.dir)
except FileExistsError:
    pass

mbp.reconnect_monomers(h2o_mbpol)
mbp.reconnect_monomers(h2o_mbpol)

h2o_mbpol.calc = mbp.MbpolCalculator(h2o_mbpol)
h2o_siesta.calc = SiestaH2O(basis = args.basis, xc = args.xc)

E0, h2o_siesta, h2o_mbpol = prop_and_eval(0, h2o_siesta, args, True)

rands = np.random.rand(args.Nmax)
temperature = args.T * ase_units.kB

trajectory = Trajectory(args.dir + 'verlet' + file_extension + '_keepv_dc.traj', 'a')
log = logger.MDLogger(dyn, h2o_siesta, args.dir + 'verlet' + file_extension + '_keepv_dc.log')
log_mbpol = logger.MDLogger(dyn, h2o_mbpol, args.dir + 'verlet' + file_extension + '_keepv_dc.mbpol.log')
log_all = logger.MDLogger(dyn, h2o_siesta, args.dir + 'verlet' + file_extension + '_keepv_dc.all.log')

trajectory.write(h2o_siesta)
log()
shuffle_momenta = False
for i in range(args.Nmax):
    id_list = list(range(n_clients))
    h2o_list = [Atoms(h2o_siesta) for i in id_list]
    args_list = [args] * n_clients
    shuffle_momenta_list = [shuffle_momenta] * n_clients
    if n_clients > 1:
        E1, h2o_siesta_new, h2o_mbpol_new = \
            view.map_sync(prop_and_eval(id_list, h2o_list, args_list,
                shuffle_momenta_list))
    else:
        E1, h2o_siesta_new, h2o_mbpol_new = \
            list(map(prop_and_eval(id_list, h2o_list, args_list,
                shuffle_momenta_list)))

    de = np.array(E1) - E0
    where_accepted = np.where(de <= 0)[0]
    if len(where_accepted) > 0:
        which = where_accepted[0]
    else:
        which = np.argmin(de)

    h2o_siesta_new = h2o_siesta_new[which]
    h2o_mbpol_new = h2o_mbpol_new[which]
    de = de[which]
    print('Energy difference: {} eV'.format(de))
    if de <= 0 or rands[i] < np.exp(-de/temperature):
        h2o_siesta = h2o_siesta_new
        h2o_mbpol = h2o_mbpol_new
        trajectory.write(h2o_siesta)
        log()
#        shuffle_momenta(h2o)
        E0  = h2o_siesta.get_total_energy()
        shuffle_momenta = False
    else:
        shuffle_momenta = True
        continue


def prop_and_eval(engine_id, h2o_siesta, args, shuffle_momenta = False):
    import numpy as np
    import mbpol_calculator as mbp
    from ase import Atoms
    from ase.md import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units as ase_units
    import shutil
    import os
    os.environ['QT_QPA_PLATFORM']='offscreen'

    try:
        os.chdir('./{}/'.format(engine_id))
    except FileNotFoundError:
        os.mkdir('./{}/'.format(engine_id))
        os.chdir('./{}/'.format(engine_id))

    if shuffle_momenta:
        MaxwellBoltzmannDistribution(h2o_siesta, args.T * ase_units.kB)
        h2o_siesta.set_momenta(h2o_siesta.get_momenta() - \
         np.mean(h2o_siesta.get_momenta(),axis =0))

    h2o_mbpol = Atoms(h2o_siesta)
    h2o_mbpol.calc = mbp.MbpolCalculator(h2o_mbpol)
    dyn = VelocityVerlet(h2o_mbpol, args.dt * ase_units.fs)
    dyn.run(args.Nt)
    h2o_siesta.set_positions(h2o_mbpol.get_positions())
    E1 = h2o_siesta.get_total_energy()
    return E1, h2o_siesta, h2o_mbpol
