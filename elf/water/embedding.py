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
