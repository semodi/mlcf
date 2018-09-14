""" Module that contains functions to preprocess quantumm chemical data such as
atomic positions and/or positions and width of Gaussians that are fitted
to the charge density. Preprocessed datasets are then used in subnet and network,
for a Neural Network similar to that proposed by Behler et al.
"""

import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Dataset containing the coordinates and width ( = 1 for atoms) for
# Gaussians and Atoms
Dataset = namedtuple("Dataset", "data species species_index n")

RadParameters = namedtuple("RadParameters", "r_s eta r_c")
AngParameters = namedtuple("AngParameters", "zeta lambd eta r_c")

def overlap(X0, X1, X2, dim=2):

    if dim == 2 :
        n = 1/(X0[:,3] + X1[:,3])
    else:
        n = 1/(X0[:,3] + X1[:,3] + X2[:,3])

    return n**(3/2) * np.exp(-n)
    # return 1

def f_c(r, r_c):
    """ Cutoff function
    """
    result = .5 * (np.cos(np.pi*r / r_c) + 1)
    result[r > r_c] = 0
    return result

def sym_r(X0, X1, r_s, eta, r_c):
    """ Radial symmetry function"""
    r = np.linalg.norm(X0[:, :3] - X1[:, :3], axis=1)
    return overlap(X0,X1,None,2)*np.exp(-eta * (r - r_s) ** 2 ) * f_c(r, r_c)

def sym_ang(X0, X1, X2, zeta, lambd, eta, r_c):
    """ Angular symmetry function"""
    r01 = np.linalg.norm(X0[:, :3] - X1[:, :3], axis=1)
    r12 = np.linalg.norm(X1[:, :3] - X2[:, :3], axis=1)
    r02 = np.linalg.norm(X0[:, :3] - X2[:, :3], axis=1)
    return overlap(X0,X1,X2,3) * np.exp(-eta * (r01 ** 2 + r12 ** 2 + r02 ** 2)) * \
        f_c(r01, r_c) * f_c(r12, r_c) * f_c(r02, r_c) * \
        (1 + lambd * np.diag(np.dot(X1[:, :3]-X0[:, :3],
            (X2[:, :3]-X0[:, :3]).T))/(r01*r02)) ** zeta

def reshape(d):
    """Reshape data from format (n_samples * n_copies, 4 or 3)
    into format (n_samples, n_copies, 4)
    """

    # Reshape data and add column of 1s for width if not Gaussian
    if d.data.shape[1] == 3:
        d = d._replace(data = np.concatenate((d.data, np.ones([len(d.data), 1])), axis=1))
    d = d._replace(data = d.data.reshape(int(len(d.data)/d.n),d.n,4))

    return d

def reshape_group(x, n):
    """Reshape data from format (n_samples, n_copies * 4)
    into format (n_copies, n_samples, 4) needed by tensorflow
    """

    n0 = x.shape[0]
    n1 = int(x.shape[1]/n)
    x = x.T.reshape(n,n1,n0).swapaxes(1,2)

    return x


def get_o_h(atoms, n_molecules):
    """ Given a np.array (n_samples * n_molecules * 3, 3)
    containing water molecule coordinates, return the oxygen
    and hydrogen datasets
    """

    n_samples = int(len(atoms)/(n_molecules*3))
    atoms = np.concatenate([atoms, np.ones([len(atoms),1])], axis = 1)
    atoms = atoms.reshape(n_samples,n_molecules*3, 4)


    oxy_list = []
    hydro_list = []
    for m in range(n_molecules):
        oxy_list.append(m*3)
        hydro_list.append(m*3+1)
        hydro_list.append(m*3+2)

    oxygen =  atoms[:,oxy_list,:].reshape(n_samples*n_molecules,4)
    hydrogen = atoms[:,hydro_list,:].reshape(n_samples*n_molecules*2,4)

    return Dataset(oxygen,'O',1,n_molecules), Dataset(hydrogen,'H',2,n_molecules*2)


def scale_together(subnets):
    """ Scale data in given subnets by combining the data ranges
        (Should always be performed if subnets share weights)
    """

    for s in subnets:
        if not s.species == subnets[0].species:
            print('Warning, subnets do not contain the same species. Proceeding...')

    if not isinstance(subnets ,list):
        Exception('Input must be a list of subnets')

    all_data = np.zeros([0,s.features])
    for i, s in enumerate(subnets):
        subnets[i].X_train= subnets[i].X_train.reshape(-1, subnets[i].features)
        subnets[i].X_train = subnets[i].scaler.inverse_transform(subnets[i].X_train)
        subnets[i].X_test= subnets[i].X_test.reshape(-1, subnets[i].features)
        subnets[i].X_test = subnets[i].scaler.inverse_transform(subnets[i].X_test)
        all_data = np.concatenate([all_data,subnets[i].X_train], axis=0)

    subnets[0].scaler.fit(all_data)

    for i,s in enumerate(subnets):
        subnets[i].scaler = subnets[0].scaler
        subnets[i].X_train = subnets[i].scaler.transform(subnets[i].X_train)
        subnets[i].X_train = subnets[i].X_train.reshape([-1, subnets[i].features*s.n_copies])
        subnets[i].X_train = reshape_group(subnets[i].X_train, subnets[i].n_copies)
        subnets[i].X_test = subnets[i].scaler.transform(subnets[i].X_test)
        subnets[i].X_test = subnets[i].X_test.reshape([-1, subnets[i].features*subnets[i].n_copies])
        subnets[i].X_test = reshape_group(subnets[i].X_test, subnets[i].n_copies)

    return subnets

def preprocess(datasets, rad_param, ang_param, which = -1, fraction = 1.0, seed = 42):
    """ Preprocess raw data, using radial and angular symmetry functions
    proposed by Behler et al.

    Parameters:
    ----------
    datasets: list of Dataset; symmetry functions are computed by 'overlaps'
        between the first Dataset and itself and the other Datasets.
        Data should be a numpy array of the shape ( ,4) or ( , dataset.n, 4).
    rad_param: RadParameters; parameters for the radial symmetry functions
    ang_param: AngParameters; parameters for the angular symmetry functions
    which; int/[int]; determines which species the overlaps are computed
        with. default: -1; overlap with all datasets

    Returns:
    --------
    X: (n_samples, n_copies, 4)- numpy.array; preprocessed data in the shape
        that is required by tensorflow
    """

    # Check if datasets are in right order and have the same length
    index = -1
    length = 0
    for i, d in enumerate(datasets):

        if len(d.data.shape) == 3:
            if not d.data.shape[1] == d.n:
                raise Exception('Invalid dataset shape {},' +
                ' expected (, dataset.n, 4)'.format(d.data.shape))
        elif d.data.ndim == 2:
            datasets[i] = reshape(d)
            d = datasets[i]
        else:
            raise Exception('Invalid dataset shape {},' +
             ' expected (, dataset.n, 4) or ( , 4)'.format(d.data.shape))

        if i == 0:
            # Number of samples
            length = int(d.data.shape[0])
        else:
            if d.species_index == index:
                raise Exception('Same species can only be contained once' +
                ' in dataset. Merge datasets that contain the same species')
            if d.species_index < index:
                raise Exception('Datasets not in the right order.' +
                ' Sort them by ascending species index!')
            index = d.species_index

        if int(d.data.shape[0]) != length :
            raise Exception('Number of samples not equal across datasets')



    if fraction < 1.0 :
        old_datasets = datasets.copy()
        datasets = []
        for d in old_datasets:
            data, _ = train_test_split(d.data,
                                    test_size = 1 - fraction,
                                    random_state = seed,
                                    shuffle = True)
            datasets.append(Dataset(data, d.species, d.species_index, d.n))

    # Preprocess data
    set0 = datasets[0]
    n_samples = len(set0.data)
    X = np.zeros([n_samples, set0.n, 1])

    if not isinstance(which,list): which = [which]

    for n0 in range(set0.n):
        ifeat = 0
        for di, set1 in enumerate(datasets):
            spec0 = set0.species_index
            spec1 = set1.species_index
            if (not spec1 in which) and (not which == [-1]): continue

            if spec1 < spec0: spec0, spec1 = spec1, spec0

            # Radial symmetry functions
            for eta in rad_param.eta[spec0,spec1,:]:
                if eta == 999: continue
                for r_s in rad_param.r_s[spec0,spec1,:]:
                    if r_s == 999: continue
                    feat = np.zeros(n_samples)
                    for n1 in range(set1.n):
                        # Skip self-"interaction"
                        if n0 == n1 and di == 0:
                            continue
                        feat += sym_r(set0.data[:,n0,:], set1.data[:,n1,:],
                                      r_s, eta, rad_param.r_c[spec0, spec1])


                    #for first iteration,
                    #determine  number of features dynamically
                    if n0 == 0:
                        X[:, n0, -1] = feat
                        X = np.concatenate((X, np.zeros([n_samples, set0.n, 1])),
                                axis = 2)
                    else:
                        X[:, n0, ifeat] = feat
                        ifeat += 1

                    if eta == 0: break

            # Angular symmetry functions
            for zeta in ang_param.zeta[spec0,spec1,:]:
                if zeta == 999: continue
                for lambd in ang_param.lambd[spec0,spec1,:]:
                    if lambd == 999: continue
                    for eta in ang_param.eta[spec0,spec1,:]:
                        if eta == 999: continue
                        feat = np.zeros(n_samples)
                        for n1 in range(set1.n):
                            for n2 in range(set1.n):

                                if (n0 == n1 or n0 == n2) and di == 0:
                                    continue
                                if n1 == n2: continue

                                feat += sym_ang(set0.data[:, n0, :],
                                                set1.data[:, n1, :],
                                                set1.data[:, n2, :],
                                                zeta, lambd, eta,
                                                ang_param.r_c[spec0, spec1])

                        feat *= 2**(1-zeta)
                        if n0 == 0:
                            X[:, n0, -1] = feat
                            X = np.concatenate((X, np.zeros([n_samples, set0.n, 1])),
                                    axis = 2)
                        else:
                            X[:, n0, ifeat] = feat
                            ifeat += 1

    dataset_new = Dataset(X[:,:,:-1],
                          datasets[0].species,
                          datasets[0].species_index,
                          datasets[0].n)

    return dataset_new
