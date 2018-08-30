import h5py
import json
import numpy as np
import elf
from ase import Atoms
from ase.io import write
import os
from elf import ElF
import ipyparallel as ipp
import re
import pandas as pd 

class serial_view():
    def __init__(self):
        pass
    def map_sync(self, *args):
        return list(map(*args))

def get_view(profile = 'default'):
    client = ipp.Client()
    view = client.load_balanced_view()
    print('Clients operating : {}'.format(len(client.ids)))
    n_clients = len(client.ids)
    return view

def __get_elfs(path, atoms, basis, method):
    density = elf.siesta.get_density(path)
    return elf.real_space.get_elfs_oriented(atoms, density, basis, method)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def preprocess_all(root, basis, dens_ext = 'RHOXC',
    add_ext = 'out', method = 'elf', view = serial_view()):

    if root[0] != '/' and root[0] != '~':
        root = os.getcwd() + '/' + root
    paths = []

    for branch in os.walk(root):
        files = np.unique([t.split('.')[0] for t in branch[2] if\
            (dens_ext in t or add_ext in t)])
        paths += [branch[0] + '/' + f for f in files]

    # Sort path for custom directory structure node_*/*.ext

    paths = sorted(paths, key=natural_keys)

    print(paths)
    print('{} systems found. Processing ...'.format(len(paths)))

    atoms = view.map_sync(elf.siesta.get_atoms,[p + '.' + add_ext for p in paths])
    elfs = view.map_sync(__get_elfs,[p + '.' + dens_ext for p in paths],
     atoms, [basis]*len(atoms), [method]*len(atoms))

    forces = view.map_sync(elf.siesta.get_forces, [p + '.' + add_ext for p in paths])
    energies = view.map_sync(elf.siesta.get_energy, [p + '.' + add_ext for p in paths])
    forces = np.array(forces).reshape(-1,3)
    energies = np.array(energies).flatten()

    name = root.split('/')[-1]
    elfs_to_hdf5(elfs, name + '_processed.hdf5')
    write(name +'.traj', atoms)
    pd.DataFrame(forces).to_csv(name + '.forces', index = None, header = None)
    pd.DataFrame(energies).to_csv(name + '.energies', index = None, header = None)
    return elfs

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
    print(basis)
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

        bas =  dict(filter(lambda x: species.astype(str).lower() in x[0].lower(), basis.items()))
        elfs[system].append(ElF(value[:length], angles, bas, species.astype(str),np.zeros(3)))

    return elfs
