from ase import Atoms
from siestah2o import Timer
from ase.calculators.siesta.siesta import SiestaTrunk462 as Siesta 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mbpol_calculator import MbpolCalculator
from ase.units import Ry, kB
import os

os.chdir('test_run')
a = 15.646 

os.environ['SIESTA_COMMAND'] = 'mpirun -n {} siesta < ./%s > ./%s'.format(os.environ['PBS_NP'])
os.environ['SIESTA_PP_PATH'] = '/gpfs/home/smdick/psf/'


h2o = Atoms('128OHH',
            positions = np.genfromtxt('128.csv',delimiter = ','),
            cell = [a, a, a],
            pbc = True)

calc = Siesta(label='H2O',
               xc= ['LDA','PW92'],
               mesh_cutoff=200 * Ry,
               energy_shift=0.02 * Ry,
               fdf_arguments={'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 50,
                              'DM.NumberPulay': 3,
                              'DM.Tolerance': 1e-1,
                              'ElectronicTemperature': 5e-3,
                              'WriteMullikenPop': 1,
                              'DM.FormattedFiles': 'True'})

h2o.set_calculator(calc)
# h2o.calc = MbpolCalculator(h2o)
timer = Timer('timer_md_test')
e = h2o.get_potential_energy()
timer.stop()
