"""Utility functions for real-space grid properties
"""
import numpy as np
import struct
from ase.units import Bohr
from elf.real_space.density import Density
from ase import Atoms
import re

def get_density_bin(file_path):
    """ Same as get_data for binary (unformatted) files
    """
    #Warning: Only works for cubic cells!!!
    #TODO: Implement for arb. cells

    bin_file = open(file_path, mode = 'rb')

    unitcell = '<I9dI'
    grid = '<I4iI'

    unitcell = np.array(struct.unpack(unitcell,
        bin_file.read(struct.calcsize(unitcell))))[1:-1].reshape(3,3)

    grid = np.array(struct.unpack(grid,bin_file.read(struct.calcsize(grid))))[1:-1]
    if (grid[0] == grid[1] == grid[2]) and grid[3] == 1:
        a = grid[0]
    else:
        raise Exception('get_data_bin cannot handle non-cubic unitcells or spin')

    block = '<' + 'I{}fI'.format(a)*a*a
    content = np.array(struct.unpack(block,bin_file.read(struct.calcsize(block))))

    rho = content.reshape(a+2, a, a, order = 'F')[1:-1,:,:]
    return Density(rho, unitcell*Bohr, grid[:3])

def get_density(file_path):
    """Import data from RHO file (or similar real-space grid files)
    Data is saved in global variables.

    Structure of RHO file:
    first three lines give the unit cell vectors
    fourth line the grid dimensions
    subsequent lines give density on grid

    Parameters:
    -----------
    file_path: string; path to RHO file from which density is read

    Returns:
    --------
    None

    Other:
    ------
    unitcell: (3,3) np.array; saves the unitcell dimension in euclidean coordinates
    grid: (,3) np.array; number of grid points in each euclidean direction
    rho: (grid[1],grid[2],grid[3]) np.array; density on grid
    """
    rhopath = file_path
    unitcell = np.zeros([3, 3])
    grid = np.zeros([4])

    with open(file_path, 'r') as rhofile:

        # unit cell (in Bohr)
        for i in range(0, 3):
            unitcell[i, :] = rhofile.readline().split()

        grid[:] = rhofile.readline().split()
        grid = grid.astype(int)
        n_el = grid[0] * grid[1] * grid[2] * grid[3]

        # initiatialize density with right shape
        rho = np.zeros(grid)

        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    rho[x, y, z, 0] = rhofile.readline()

    # closed shell -> we don't care about spin.
    rho = rho[:, :, :, 0]
    grid = grid[:3]
    return Density(rho, unitcell*Bohr, grid)

def get_energy(path, keywords=['Total']):
    """find energy values specified by keywords
    in siesta output file.
    """

    assert isinstance(keywords, (list, tuple))
    values = []
    with open(path, 'r') as file:
        for keyword in keywords:
            file.seek(0)
            p = re.compile('siesta:.*' + keyword + ' =.*-?\d*.?\d*')
            p_wo = re.compile('siesta:.*' + keyword + ' =\s*')
            content = file.read()
            with_number = p.findall(content)[0]
            wo_number = p_wo.findall(content)[0]

            values.append((float)(with_number[len(wo_number):]))

    return values

def get_forces(path):
    with open(path, 'r') as infile:
        infile.seek(0)

        p = re.compile("siesta: Atomic forces \(eV/Ang\):\nsiesta:.*siesta:    Tot ", re.DOTALL)
        p2 = re.compile(" 1 .*siesta: -", re.DOTALL)
        alltext = p.findall(infile.read())
        alltext = p2.findall(alltext[0])
        alltext = alltext[0][:-len('\nsiesta: -')]
        forces = []
        for i, f in enumerate(alltext.split()):
            if i%5 ==0: continue
            if f =='siesta:': continue
            forces.append(float(f))
    return np.array(forces).reshape(-1,3)


def get_atoms(path):
    def find_coords(path):
        with open(path, 'r') as infile:
            infile.seek(0)

            p = re.compile("%block AtomicCoordinatesAndAtomicSpecies.*%endblock AtomicCoordinatesAndAtomicSpecies",
                           re.DOTALL)
            alltext = p.findall(infile.read())
            return(np.array(alltext[0].split()[2:-2]).reshape(-1,4).astype(float))

    def find_chem_species(path):
        with open(path, 'r') as infile:
            infile.seek(0)

            p = re.compile("%block ChemicalSpeciesLabel.*?%endblock ChemicalSpeciesLabel",
                           re.DOTALL)
            alltext = p.findall(infile.read())
            return(np.array(alltext[0].split()[2:-2]).reshape(-1,3))

    def find_unit_cell(path):
        with open(path, 'r') as infile:
            infile.seek(0)
            p = re.compile("%block LatticeVectors.*?%endblock LatticeVectors",
                           re.DOTALL)
            alltext = p.findall(infile.read())
            lattice_vec = np.array(alltext[0].split()[2:-2]).reshape(-1,3).astype(float)
            infile.seek(0)
            p = re.compile("LatticeConstant\s*\d*\.\d*",
                           re.DOTALL)
            alltext = p.findall(infile.read())
            a = alltext[0]
            a = np.array(re.compile("\d+\.\d+").findall(a)[0]).astype(float)
            return lattice_vec * a

    chem_species_array = find_chem_species(path)
    chem_species = {}
    for c in chem_species_array:
        chem_species[int(c[0])] = c[2]

    atom_string = ''
    coords = find_coords(path)
    for c in coords:
        atom_string += chem_species[int(c[3])]
    atoms = Atoms(atom_string, positions = coords[:,:3])
    atoms.set_pbc(True)
    atoms.set_cell(find_unit_cell(path))
    return atoms
