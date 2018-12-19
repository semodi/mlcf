from .calculator import *
import ipyparallel as ipp
from .read_input import read_input
from mlc_func.elf import serial_view
import os
import shutil



def single_thread(atoms_list, calcfile, env):
    import os
    from mlc_func.calculator import read_input
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.environ['SIESTA_COMMAND'] = env['SIESTA_COMMAND']
    os.environ['SIESTA_PP_PATH'] = env['SIESTA_PP_PATH']
    os.environ['SIESTA_WORKING_DIR'] = env['SIESTA_WORKING_DIR']
    os.chdir(env['SIESTA_WORKING_DIR'])
    calculator = load_from_file(calcfile)
    results = []

    for atoms in atoms_list:
        sys_idx, atoms = list(atoms.items())[0]
        try:
            shutil.os.mkdir('{}'.format(sys_idx))
        except FileExistsError:
            pass
        os.chdir('{}'.format(sys_idx))
        atoms.calc = calculator
        pe = atoms.get_potential_energy()
        forces = atoms.get_forces()
        results.append({sys_idx: [pe, forces]})
        os.chdir('../')
    return results

class ListIntegrator():

    def __init__(self, atoms, calcfile):
        """This is not really an ASE integrator (but it can be used like one). Instead
        of integrating the equations of motion it simply loops through a set of configurations.
        It can therefore be used to generate datasets that are then used to train MLCFs.

        Parameters
        -----------
        atoms: list of ase.Atoms
            configurations to calculate
        calcfile: str,
            path to file that defines the calculator which should be used
        """
        settings, _ = read_input(calcfile)
        self.atoms = atoms
        self.calcfile = calcfile
        if settings['ipp_client'] != None:
            client = ipp.Client(profile=settings['ipp_client'])
            self.view = client.load_balanced_view()
        else:
            self.view = serial_view()

        atoms_indexed = [{i: a} for i, a in enumerate(atoms)]
        print(atoms_indexed)
        n_workers = len(self.view)

        self.atoms_threaded = [atoms_indexed[n::n_workers] for n in range(n_workers)]

    def run(self, _):
        """
        Run the ListIntegrator until all configurations are calculated
        """
        env = dict(os.environ)
        self.view.map_sync(single_thread, self.atoms_threaded,
                [self.calcfile]*len(self.atoms_threaded),
                [env]*len(self.atoms_threaded))
