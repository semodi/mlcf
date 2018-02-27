from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.units import Ry

a = 20

# =======  QZDP - 8.5 =========

o_basis = """ 3
n=2 0 4 E 50. 7.5
    8.0 5.0 3.5 2.0
n=2 1 4 E 10. 8.3
    8.5 5.0 3.5 2.0
n=3 2 2 E 40. 8.3 Q 6.
    8.5 2.2"""

h_basis = """ 2
n=1 0 4 E 50. 8.3
    8.5 5.0 3.5 2.0
n=2 1 2 E 20. 7.8 Q 3.5
    8.0 2.0"""

species_o = Species(symbol= 'O', basis_set= PAOBasisBlock(o_basis))
species_h = Species(symbol= 'H', basis_set= PAOBasisBlock(h_basis))


h2o = Atoms('OHH',
            positions = [[0,0,0],[0.757,0.586,0],[-0.757, 0.586,0]],
            cell = [a, a, a],
            pbc = True)

calc = Siesta(label='H2O',
               xc='PBE',
               mesh_cutoff=300 * Ry,
               species=[species_o, species_h],
               fdf_arguments={'DM.MixingWeight': 0.3,
                              'MaxSCFIterations': 100,
                              'DM.NumberPulay': 3,
                              'DM.Tolerance': 5e-5})
h2o.set_calculator(calc)
e = h2o.get_potential_energy()
