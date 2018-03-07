from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta 
import numpy as np
from ase import Atoms
from ase.io import read,write, Trajectory
import shutil
import pimd_wrapper as pimd
import os
import ipyparallel as ipp
import time
from ase.units import Ry
#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['SIESTA_COMMAND'] = 'siesta < ./%s > ./%s'
os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'

PPN = 16

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


def prop_and_eval(engine_id):
    from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta 
    import numpy as np
    import sys
    sys.path.append('/gpfs/home/smdick/md_routines')
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units as ase_units
    from ase.units import Ry
    import shutil
    import os
    from ase.io import Trajectory
    os.environ['SIESTA_COMMAND'] = 'mpiexec -np 16 -f machinefile siesta < ./%s > ./%s'
    os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.chdir('/gpfs/home/smdick/md_routines/siesta/{}/'.format(engine_id))
    
    h2o_siesta = Trajectory('tmp.traj','r')[0] 
    h2o_siesta.calc = Siesta(label='H2O',
               xc= ['VDW','BH'],
               mesh_cutoff=200 * Ry,
               energy_shift=0.02 * Ry,
               fdf_arguments={'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 50,
                              'DM.NumberPulay': 3,
                              'DM.Tolerance': 5e-5,
                              'ElectronicTemperature': 5e-3,
                              'WriteMullikenPop': 1,
                              'DM.FormattedFiles': 'True'})
    E1 = h2o_siesta.get_total_energy()
    os.chdir('../')
    return E1

if __name__ == '__main__':
    os.chdir('/gpfs/home/smdick/md_routines')
    
    with open(os.environ['PBS_NODEFILE'],'r') as nodefile:
        hosts = nodefile.read().split()
    print(hosts)
    nodes = np.unique(hosts)
    
    # Load initial configuration
    b = 13.0
    h2o_siesta = Atoms('64OHH',
                positions = read('64.xyz').get_positions(),
                cell = [b, b, b],
                pbc = True)

    try:
        os.chdir('./siesta/')
    except FileNotFoundError:
        os.mkdir('./siesta/')
        os.chdir('./siesta/')

    for i in range(n_clients):
        try:
            shutil.os.mkdir('{}'.format(i))
        except FileExistsError:
            pass
        with open('{}/machinefile'.format(i), 'w') as machinefile:
            for rep in range(PPN):
                machinefile.write('{}\n'.format(nodes[i]))

    h2o_siesta.calc = Siesta(label='H2O',
               xc= ['VDW','BH'],
               mesh_cutoff=200 * Ry,
               energy_shift=0.02 * Ry,
               fdf_arguments={'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 50,
                              'DM.NumberPulay': 3,
                              'DM.Tolerance': 5e-5,
                              'ElectronicTemperature': 5e-3,
                              'WriteMullikenPop': 1,
                              'DM.FormattedFiles': 'True'})



    for i in range(n_clients):
        traj = Trajectory('./{}/tmp.traj'.format(i),'w')
        traj.write(h2o_siesta)

    id_list = list(range(n_clients))

    if n_clients > 1:
        E1 = \
            view.map_sync(prop_and_eval,id_list)
    else:
        E1 = \
           list(map(prop_and_eval, id_list))




