import numpy as np
import spherical_functions as sf
from sympy.physics.wigner import wigner_3j
from sympy import N

# Transformation matrix between radial and euclidean (real) representation of
# a rank-1 tensor
T = np.array([[1j,0,1j], [0,np.sqrt(2),0], [1,0,-1]]) * 1/np.sqrt(2)

def get_max(tensor):
    """
    Get the maximum radial index and maximum ang. momentum in tensor
    """
    for n in range(1000):
        if not '{},0,0'.format(n) in tensor:
            n_max = n
            break
    for l in range(1000):
        if not '0,{},0'.format(l) in tensor:
            l_max = l
            break
    return n_max, l_max

def make_real(tensor):
    """
    Take complex tensors tensor provided as a dict and convert them into
    real tensors
    """
    tensor_real = []
    n_max, l_max = get_max(tensor)
    for n in range(n_max):
        for l in range(l_max):
            for m in range(-l,0):
                tensor_real.append((-1j/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]-(-1)**m*tensor['{},{},{}'.format(n,l,-m)])).real)
            tensor_real.append(tensor['{},{},{}'.format(n,l,0)].real)
            for m in range(1,l+1):
                tensor_real.append((-1/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,-m)]+(-1)**m*tensor['{},{},{}'.format(n,l,m)])).real)

    return np.array(tensor_real)

def make_real_old(tensor):
    """
    Take complex tensors tensor provided as a dict and convert them into
    real tensors (old version as used by xcml)
    """
    tensor_real = []
    n_max, l_max = get_max(tensor)
    for n in range(n_max):
        for l in range(l_max):
            for m in range(-l,0):
                tensor_real.append((-1/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]+(-1)**m*tensor['{},{},{}'.format(n,l,-m)])).real)
                tensor_real.append((-1j/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]-(-1)**m*tensor['{},{},{}'.format(n,l,-m)])).real)
#                assert np.allclose((-1/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]+(-1)**m*tensor['{},{},{}'.format(n,l,-m)])).imag,0)

            tensor_real.append(tensor['{},{},{}'.format(n,l,0)].real)
#            assert np.allclose(tensor['{},{},{}'.format(n,l,0)].imag, 0)

    return np.array(tensor_real)

def get_casimir(tensor):
    """ Get the casimir element (equiv. to L_2 norm) of a tensor
    """
    casimir = {}

    n_max, l_max = get_max(tensor)
    for n in range(n_max):
        for l in range(l_max):
            if not '{},{},0'.format(n,l) in tensor:
                break
            casimir['{},{}'.format(n,l)] = 0
            for m in range(-l, l+1):
                casimir['{},{}'.format(n,l)] += np.abs(tensor['{},{},{}'.format(n,l,m)])**2
    return casimir


def get_euler_angles(co):
    """ Given a coordinate system co, return the euler angles
    that relate this CS to the standard CS

    Parameters:
    -----------------
    co: np.ndarray (3,3); coordinates of the body-fixed axes in the
                          global coordinate system

    Returns:
    -------
    alpha, beta, gamma: float; euler angles

    """
    beta = np.arccos(co[2,2])
    alpha = np.arctan2(co[2,1], co[2,0])

    g = np.round(np.sin(alpha) * co[1,0] - np.cos(alpha) * co[1,1],10)
    N = np.array([-np.sin(alpha),np.cos(alpha),0])
    prefac = np.cross(co[2],N).dot(co[1])
    if prefac == 0:
        prefac = N.dot(co[1])

    prefac /= abs(prefac)

    gamma = np.arctan2(prefac * np.sqrt(1 - g**2), -g)

    return alpha, beta, gamma


def rotate_vector(vec, angles, inverse = False):
    """ Rotate a real vector (euclidean order: xyz) with euler angles
        inverse = False: rotate vector
        inverse = True: rotate CS"""
    vec = vec[:,[1,2,0]]
    T_inv = np.conj(T.T)

    R = T.dot(sf.Wigner_D_element(*angles,np.array([1])).reshape(3,3).dot(T_inv))
    assert np.allclose(R.conj(), R)

    if inverse:
        vec = np.einsum('ij,kj -> ki', R.T, vec)
    else:
        vec= np.einsum('ij,kj -> ki', R, vec)

    return vec[:,[2,0,1]].real

def rotate_tensor(tensor, angles, inverse = False):
    """ Rotate a (complex!) tensor.

    Parameters:
    ----------
    tensor: dict; rank-2 tensor to rotate; the tensor is expected to be complete
        that is no entries should be missing
    angles: euler angles, {alpha, beta, gamma}
    inverse: boolen; inverse rotation

    Returns:
    ---------
    Rotated version of vec

    Info:
    -------
    Remember that in nncs and elfcs alignment, inverse = True should be used
    """
    if not isinstance(tensor['0,0,0'], np.complex128) and not isinstance(tensor['0,0,0'], np.complex64)\
        and not type(tensor['0,0,0']) == complex:
        raise Exception('tensor has to be complex')
    R = {}

    n_max, l_max = get_max(tensor)
    for l in range(1,l_max):
        if not '0,{},0'.format(l) in tensor:
            break
        R[l] = sf.Wigner_D_element(*angles,np.array([l])).reshape(2*l+1,2*l+1)

    tensor_rotated = {}
    for n in range(n_max):
        if not '{},0,0'.format(n) in tensor:
            break

        tensor_rotated['{},0,0'.format(n)] = tensor['{},0,0'.format(n)]
        for l in range(1, l_max):
            if not '0,{},0'.format(l) in tensor:
                break
            t = []
            for m in range(-l,l+1):
                t.append(tensor['{},{},{}'.format(n,l,m)])
            t = np.array(t)
            if inverse:
                t_rotated = R[l].T.conj().dot(t)
            else:
                t_rotated = R[l].dot(t)
            for m in range(-l,l+1):
                tensor_rotated['{},{},{}'.format(n,l,m)] = t_rotated[l+m]
    return tensor_rotated


def get_P(tensor, wig3j = None):
    P = []
    n_rad, n_l = get_max(tensor)

    lam = 1
    # It is faster to pre-evaluate the wigner-3j symbol, even faster if it is passed
    if not isinstance(wig3j, np.ndarray):
        wig3j = np.zeros([n_l,n_l,2*n_l+1,2*n_l+1,2*n_l+1])
        wig3j = wig3j.astype(np.complex128)
        for l1 in range(n_l):
            for l2 in range(n_l):
                for m in range(-lam,lam+1):
                    for m1 in range(-n_l,n_l+1):
                        for m2 in range(-n_l,n_l+1):
                            wig3j[l2,l1,m,m1,m2] = N(wigner_3j(lam,l2,l1,m,m1,m2))


    for mu in range(-lam,lam + 1):
        P.append([])
        for n1 in range(n_rad):
            for n2 in range(n_rad):
                for l1 in range(n_l):
                    for l2 in range(n_l):
                        if (l1 + l2)%2 == 0: continue
                        p = 0
                        for m in range(-n_l, n_l+1):
                            wig = wig3j[l2,l1,mu,(m-mu),-m]
                            if wig != 0:
                                p += tensor['{},{},{}'.format(n1,l1,m)]*tensor['{},{},{}'.format(n2,l2,m-mu)].conj() *\
                                  (-1)**m * wig
                        p *= (-1)**(lam-l2)
                        P[mu+lam].append(p)
    return P

def tensor_to_P(tensor):
    """
    Transform an arbitray SO(3) tensor into P which transforms under the irreducible
    representation with l = 1
    """
    p = np.array(get_P(tensor))
    p_real = []
    for pi in np.array(p).T:
        p_real.append(T.dot(pi)[[2,0,1]])
    p = np.array(p_real).T
    if not np.allclose(p.imag,np.zeros_like(p)):
        raise Exception('Ooops, something went wrong. P not purely real.')
    return p.real.T

def get_elfcs_angles(i, coords, tensor):
    """Use the ElF algorithm to get angles relating global to local CS
    """
    norm = np.linalg.norm
    p = tensor_to_P(tensor)[:,[2,0,1]] # Go from tensor to euclidean ordering
    axis1 = p[0]/norm(p[0])

    for d in p[1:]:
        if 1 - abs(np.dot(axis1,d)/(norm(axis1)*norm(d))) > 1e-3:
            axis2 = d
            break
    else:
        print('Using coordinates for alignment')
        c = np.array(coords[i])
        coords = np.delete(coords, i, axis = 0)
        dr = norm((coords - c), axis =1)
        order = np.argsort(dr)
        for o in order:
            axis2 = coords[o] - c
            if 1 - np.abs(axis2.dot(axis1)/norm(axis2)) > 1e-5:
                break
        else:
            raise Exception('Could not determine orientation. Aborting...')

    axis3 = np.cross(axis1, axis2)
    axis3 = axis3/norm(axis3)
    axis2 = np.cross(axis3, axis1)
    axis2 = axis2/norm(axis2)

    axis1 = axis1.round(10)
    axis2 = axis2.round(10)
    axis3 = axis3.round(10)

    angles = get_euler_angles(np.array([axis1, axis2, axis3]))
    return angles

def get_nncs_angles(i, coords, tensor = None):
    """ Get euler angles to rotate to the local CS for coords[i] that is
     oriented according to the nearest neighbor rule
    """

    norm = np.linalg.norm
    c = np.array(coords[i])
    coords_sorted = np.array(coords)
    coords_sorted = np.delete(coords_sorted, i , axis = 0)
    order = np.argsort(np.linalg.norm(coords_sorted - c, axis = 1))
    coords_sorted = coords_sorted[order]

    axis1 = coords_sorted[0] - c
    axis1 = axis1/norm(axis1)

    for u, cs in enumerate(coords_sorted[1:]):
        axis2 = cs - c
        if 1 - np.abs(axis2.dot(axis1)/norm(axis2)) > 1e-5:
            # print(order)
            # print("i = {}, Using {}".format(i, order[u+1]))
            break
    else:
        raise Exception('Could not determine orientation. Aborting...')

    axis3 = np.cross(axis1, axis2)
    axis3 /= norm(axis3)
    axis2 = np.cross(axis3, axis1)
    axis2 /= norm(axis2)

    axis1 = axis1.round(10)
    axis2 = axis2.round(10)
    axis3 = axis3.round(10)

    angles = get_euler_angles(np.array([axis1, axis2, axis3]))

    return angles
