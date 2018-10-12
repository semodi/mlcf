import h5py
import json
import numpy as np
from ase import Atoms
from ase.io import write
import os
from .elf import ElF
import ipyparallel as ipp
import re
import pandas as pd
from mlc_func.elf.siesta import get_density, get_density_bin, get_atoms, get_forces, get_energy
from mlc_func.elf.real_space import get_elfs_oriented
from .serial_view import serial_view

def get_view(profile = 'default', n = -1):
    client = ipp.Client()
    # view = client.load_balanced_view()
    if n == -1:
        view = client[:]
    else:
        view = client[:n]
    print('Clients operating : {}'.format(len(client.ids)))
    n_clients = len(client.ids)
    return view

def __get_elfs(path, atoms, basis, method):
    try:
        density = get_density(path)
    except UnicodeDecodeError:
        density = get_density_bin(path)
    return get_elfs_oriented(atoms, density, basis, method)


def __get_all(paths, method, basis, add_ext, dens_ext):
    atoms = list(map(get_atoms,[p + '.' + add_ext for p in paths]))
    elfs = list(map(__get_elfs, [p + '.' + dens_ext for p in paths],
     atoms, [basis]*len(paths), [method]*len(paths)))

    forces = list(map(get_forces, [p + '.' + add_ext for p in paths]))
    energies = list(map(get_energy, [p + '.' + add_ext for p in paths]))

    return atoms, elfs, forces, energies

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def preprocess_all(root, basis, dens_ext = 'RHOXC',
    add_ext = 'out', method = 'elf', view = serial_view()):


    if root[-1] == '/': root = root[:-1]
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

    n_workers = len(view)
    full_workload = len(paths)

    min_workload = np.floor(full_workload/n_workers).astype(int)
    max_workload = min_workload + 1
    n_max_workers = full_workload - min_workload*n_workers


    paths_dist = [paths[i*(max_workload):(i+1)*(max_workload)] for i in range(n_max_workers)]

    offset = n_max_workers*max_workload

    paths_dist += [paths[offset + i*min_workload:offset + (i+1) * min_workload] for i in range(n_workers - n_max_workers)]

    all_results = list(view.map(__get_all,
     paths_dist, [method]*len(paths_dist), [basis]*len(paths_dist),
      [add_ext]*len(paths_dist), [dens_ext]*len(paths_dist)))
    atoms, elfs, forces, energies = list(map(list, zip(*all_results)))

    forces = [e for sublist in forces for e in sublist]
    energies = [e for sublist in energies for e in sublist]
    forces = np.array(forces).reshape(-1,3)
    energies = np.array(energies).flatten()

    elfs = [e for sublist in elfs for e in sublist]
    atoms = [a for sublist in atoms for a in sublist]
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

def hdf5_to_elfs(path, species_filter = '', grouped = False,
        values_only = False, angles_only = False):

    file = h5py.File(path, 'r')
    basis = json.loads(file.attrs['basis'])
    print(basis)
    if values_only and angles_only:
        raise Exception('Cannot return values and angles only at the same time')
    if values_only:
        grouped = True

    if grouped:
        elfs_grouped = {}
    else:
        elfs = []

    if grouped:
        current_system_grouped = {}
    else:
        current_system = -1

    for value, length, species, angles, system in zip(file['value'],
                                                  file['length'],
                                                  file['species'],
                                                  file['angles'],
                                                  file['system']):
        species = species.astype(str).lower()
        if len(species_filter) > 0 and\
         (not (species in species_filter.lower())):
            continue

        if grouped:
            if not species in elfs_grouped:
                elfs_grouped[species] = []
                current_system_grouped[species] = -1
            elfs = elfs_grouped[species]
            current_system = current_system_grouped[species]

        if system != current_system:
            elfs.append([])
            if grouped:
                current_system_grouped[species] = system
            else:
                current_system = system

        bas =  dict(filter(lambda x: species in x[0].lower(), basis.items()))

        if values_only:
            elfs[system].append(value[:length])
        elif angles_only:
            elfs[system].append(angles)
        else:
            elfs[system].append(ElF(value[:length], angles, bas, species, np.zeros(3)))

    if grouped:
        elfs = elfs_grouped
    return elfs
