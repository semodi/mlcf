import argparse
from ase.io import read
import numpy as np
import ipyparallel as ipp
import matplotlib.pyplot as plt
import pandas as pd

def get_rdf(atoms, mode):
    import numpy as np
    n_mol = int(len(atoms[0].get_positions())/3)
    relevant_distances = []
    mode_shifts = {'oo':[[0,0]],'oh': [[0,1],[0,2]],'hh': [[1,1],[2,2],[1,2]]}

    norm_cnt = 0
    for atom in atoms:
        all_distances = atom.get_all_distances(mic=True)
        for i, j in mode_shifts[mode]:
            if i==j: triu_offset = 1
            else: triu_offset = 0
            relevant_distances.append(all_distances[i::3,j::3][np.triu_indices(n_mol,triu_offset)].flatten())
            norm_cnt += len(relevant_distances[-1])

    relevant_distances = np.concatenate(relevant_distances)

    bins = np.linspace(0,atoms[0].get_cell()[0,0]/2,400)
    dr = bins[1] - bins[0]
    rdf = np.zeros(len(bins) - 1)

    rdf = np.histogram(relevant_distances, bins)[0]
    g_avg = 1/(atoms[0].get_cell()[0,0]**3)
    n_obs = len(atoms)
    norm = bins[:-1]**2 * g_avg * norm_cnt * 4 * np.pi * dr
    rdf = rdf/norm

    return np.nan_to_num(rdf)

def get_bins(atoms):
    bins = np.linspace(0,atoms[0].get_cell()[0,0]/2,400)
    return bins[:-1]

def get_rdf_distributed(atoms, mode, view):
    n_clients = len(view)
    atom_blocks = [atoms[i::n_clients] for i in range(n_clients)]
    bins = get_bins(atoms)
    return bins, np.mean(np.array(view.map_sync(get_rdf, atom_blocks, [mode]*n_clients)),axis = 0)


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Create RDFS from xyz/traj file')
    parser.add_argument('file', action='store', help ='Path to .xyz/.traj file containing trajectory')
    parser.add_argument('slices', action='store', help ='Slices')
    parser.add_argument('mode', action='store', help='oo/oh/hh')
    parser.add_argument('-client', metavar='client', type=str, nargs = '?', default='none',
        help='ipp client')

    args = parser.parse_args()
    atoms = read(args.file, args.slices)
    if not args.client.lower():
        client = ipp.Client(profile = args.client)
        view = client.load_balanced_view()
        bins, rdf = get_rdf_distributed(atoms, args.mode, view)
    else:
        bins, rdf = get_bins(atoms), get_rdf(atoms, args.mode)
    results = np.zeros([len(bins),7])
    results[:,0] = bins
    results[:,5] = rdf
    pd.DataFrame(results).to_csv(args.file[::-1].split('.',1)[1][::-1] +\
     '_out_' + args.mode +'.rdf', index = None, header = None, sep='\t')

    plt.plot(bins, rdf)
    plt.ylim(0,min(np.max(rdf)+0.5,5))
    plt.show()
