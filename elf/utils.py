import h5py
import json
import numpy as np
import elf
from ase import Atoms
from ase.io import write
import os
from elf import ElF

def preprocess_all(root, basis, dens_ext = 'RHOXC',
    add_ext = 'out', method = 'elf'):

    paths = []
    for branch in os.walk(root):
        files = np.unique([t.split('.')[0] for t in branch[2]])
        paths += [branch[0] + '/' + f for f in files]

    atoms = list(map(elf.siesta.get_atoms,[p + '.' + add_ext for p in paths]))
    densities = list(map(elf.siesta.get_density, [p + '.' + dens_ext for p in paths]))
    elfs = list(map(elf.real_space.get_elfs, atoms, densities, [basis]*len(atoms)))
    oriented = list(map(elf.real_space.orient_elfs, elfs, atoms, [method]*len(atoms)))
    elfs_to_hdf5(oriented, root + '_processed.hdf5')
    write(root +'.traj', atoms)
    return oriented

def elfs_to_hdf5(elfs, path):

    # TODO: Option to check for consistency across systems
    file = h5py.File(path, 'w')
    # Find max. length for zero padding and construct
    # full basis out of atomic parts
    max_len = 0
    full_basis = {}
    system_label = ''
    for atom in elfs[0]:
        max_len = max([max_len, len(atom.value)])
        system_label += atom.species
        for b in atom.basis:
            full_basis[b] = atom.basis[b]

    file.attrs['basis'] = json.dumps(full_basis)
    file.attrs['system'] = system_label
    values = []
    lengths = []
    species = []
    angles = []
    systems = []

    for s, system in enumerate(elfs):
        for a, atom in enumerate(system):
            v = atom.value
            lengths.append(len(v))
            if len(v) != max_len:
                v_long = np.zeros(max_len)
                v_long[:len(v)] = v
                v = v_long
            values.append(v)
            angles.append(atom.angles)
            species.append(atom.species.encode('ascii','ignore'))
            systems.append(s)

    file['value'] = np.array(values)
    file['length'] = np.array(lengths)
    file['species'] = species
    file['angles'] = np.array(angles)
    file['system'] = np.array(systems)
    file.flush()
    
def hdf5_to_elfs(path, species_filter = ''):
    file = h5py.File(path, 'r')
    basis = json.loads(file.attrs['basis'])

    elfs = []
    current_system = -1
    for value, length, species, angles, system in zip(file['value'],
                                                  file['length'],
                                                  file['species'],
                                                  file['angles'],
                                                  file['system']):
        if len(species_filter) > 0 and\
         (not (species.astype(str).lower() in species_filter.lower())):
            continue

        if system != current_system:
            elfs.append([])
            current_system = system

        bas =  dict(filter(lambda x: str(species) in x[0], basis.items()))
        elfs[system].append(ElF(value[:length], angles, bas, species.astype(str)))

    return elfs
