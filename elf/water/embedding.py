import numpy as np
from sklearn.neighbors import NearestNeighbors
from ase.io import read, write

#tip4p/2005 parameters
r_oh = 0.9572
ang = 104.52 * np.pi/180
qh = -0.5564
qm = -2*qh
r_om = 0.1546

def waterc_to_tip4p(coords):
    norm = np.linalg.norm
    if coords.ndim == 2:
        coords = coords.reshape(-1,3,3)

    oh1 = coords[:,1,:] - coords[:,0,:]
    oh2 = coords[:,2,:] - coords[:,0,:]
    oh1 = oh1/norm(oh1, axis = -1).reshape(-1,1)
    oh2 = oh2/norm(oh2, axis = -1).reshape(-1,1)
    bisec = (oh1 + oh2)
    bisec = bisec/norm(bisec, axis = -1).reshape(-1,1)
    m = coords[:,0] + bisec*r_om

    # Find molecular plane

    axis1 = np.cross(bisec,oh1)
    axis1 /= norm(axis1, axis = -1).reshape(-1,1)
    axis2 = np.cross(bisec, axis1)
    assert np.allclose(norm(axis2, axis = -1), np.zeros(len(axis2)) + 1)


    h2 = np.cos(ang/2)*bisec + np.sin(ang/2)*axis2 + coords[:,0]
    h1 = np.cos(ang/2)*bisec - np.sin(ang/2)*axis2 + coords[:,0]
    h1 = np.concatenate([h1,np.zeros(len(h1)).reshape(-1,1) + qh], axis = -1)
    h2 = np.concatenate([h2,np.zeros(len(h1)).reshape(-1,1) + qh], axis = -1)
    m = np.concatenate([m,np.zeros(len(m)).reshape(-1,1) + qm], axis = -1)
    return np.concatenate([m,h1,h2], axis =  -1).reshape(-1,3,4)

def tip4p_to_str(arr):

    siesta_str = '%block Geometry.Charge \n'

    # M
    n_mol = len(arr)
    siesta_str += 'coords {} \n exp 0.05 0.15 Ang \n {} spheres \n'.format(arr[0,0,3]*n_mol, n_mol)
    for c in arr[:,0,:3]:
        siesta_str += ' {:5.4f} {:5.4f} {:5.4f} Ang \n'.format(*c)

    # H
    siesta_str += 'coords {} \n exp 0.05 0.15 Ang \n {} spheres \n'.format(arr[0,1,3]*2*n_mol, 2*n_mol)
    for c in arr[:,1:,:3].reshape(-1,3):
        siesta_str += ' {:5.4f} {:5.4f} {:5.4f} Ang \n'.format(*c)

    siesta_str += '%endblock Geometry.Charge \n'
    return siesta_str


def sample_dimers(coords, roo, epsilon = 0.1):
    if not coords.ndim == 3 or not coords.shape[1:] == (3,3):
        raise Exception('coords.shape must be (?, 3, 3)')

    coords = coords[:,0]
    found = False
    counter = 0
    while not found:
        counter += 1
        seed = np.random.randint(0,len(coords))
        seed_O = coords[seed]
        nn = NearestNeighbors(8)
        nn.fit(coords)
        dist, ind = nn.kneighbors([seed_O])
        check = (dist > roo-epsilon)& (dist < roo+epsilon)
        if counter > 200:
            return -1,-1
        if np.sum(check):
            break
    where = np.where(check)[1][0]
    print(np.linalg.norm((coords[seed] - coords[ind[0,where]])))
    return seed, ind[0,where]

r_lp = 0.7
rc_hb = 2.0

def is_hbonded(coords, ucell):

    if not coords.shape == (2,3,3):
        raise Exception('Must provide two molecules in np.array of shape (2,3,3)')

    coords = elf.geom.fold_back_coords(0, coords, ucell).reshape(-1,3,3)
    oh1 = coords[:,1,:] - coords[:,0,:]
    oh2 = coords[:,2,:] - coords[:,0,:]
    oh1 = oh1/norm(oh1, axis = -1).reshape(-1,1)
    oh2 = oh2/norm(oh2, axis = -1).reshape(-1,1)
    ohangle = np.arccos(np.einsum('ik,ik -> i',oh1,oh2))
    bisec = (oh1 + oh2)
    bisec = bisec/norm(bisec, axis = -1).reshape(-1,1)
    ortho = np.cross(oh1,oh2, axis = -1)
    lpangle = np.zeros_like(ohangle)
    lpangle[ohangle < .5 * np.pi] = (2*np.arccos(1/np.sqrt(2-1/(2-1/np.cos(ohangle/2)**2))))[ohangle < .5 * np.pi]
    lpangle[ohangle >= .5 * np.pi] = np.pi
    lpangle = lpangle.reshape(-1,1)
    lp1 = (-bisec*np.cos(lpangle/2) + ortho*np.sin(lpangle/2))*r_lp + coords[:,0,:]
    lp2 = (-bisec*np.cos(lpangle/2) - ortho*np.sin(lpangle/2))*r_lp + coords[:,0,:]

    coords = np.concatenate([coords,lp1.reshape(-1,1,3),lp2.reshape(-1,1,3)], axis = 1)

    look_at = {0: [], 1 : [3,4], 2: [3, 4], 3 : [1, 2], 4: [1, 2]}

    for i, c1 in enumerate(coords[0]):
        for c2 in coords[1,look_at[i]]:
            if np.linalg.norm(c1 - c2) <= rc_hb:
                return True

    return False

def sample_cluster(i, coords, k,  ucell, n_hb= -1):
    if n_hb == -1: n_hb = k -1

    if n_hb > k-1:
        n_hb = k - 1
        print('n_hb too large, setting to {}'.format(k-1))

    if not coords.ndim == 3 or not coords.shape[1:] == (3,3):
        raise Exception('coords.shape must be (?, 3, 3)')

    coords = elf.geom.fold_back_coords(i, coords, ucell).reshape(-1,3,3)
    coords_o = coords[:,0]
    seed_O = coords_o[i]

    nn = NearestNeighbors(2*k)
    nn.fit(coords_o)
    dist, ind = nn.kneighbors([seed_O])
    ind = ind[0]

    hbond_ind = []
    non_hbond_ind = []

    for u in ind[1:]:
        if is_hbonded(coords[[i,u]],ucell):
            hbond_ind.append(u)
        else:
            non_hbond_ind.append(u)
    if len(hbond_ind) < n_hb:
        raise Exception('Desired number of h-bonds not obtained')

    indices = [i] + hbond_ind[:n_hb]
    if len(indices) != k:
        indices += non_hbond_ind[:k-len(indices)]

    return indices

def sample_k_neighbors(coords, k):
    if not coords.ndim == 3 or not coords.shape[1:] == (3,3):
        raise Exception('coords.shape must be (?, 3, 3)')

    coords = coords[:,0]
    seed = np.random.randint(0,len(coords))
    seed_O = coords[seed]
    nn = NearestNeighbors(8)
    nn.fit(coords)
    dist, ind = nn.kneighbors([seed_O])
    return ind[0,:k]
