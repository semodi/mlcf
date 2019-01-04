""" Module that provides algebraic operations on SO(3) tensors
"""
import numpy as np
import spherical_functions as sf
from sympy.physics.wigner import wigner_3j
from sympy import N

# Transformation matrix between radial and euclidean (real) representation of
# a rank-1 tensor
T = np.array([[1j,0,1j], [0,np.sqrt(2),0], [1,0,-1]]) * 1/np.sqrt(2)
ANGLE_THRESHOLD = 1e-6
NORM_THRESHOLD = 1e-3
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
    Take complex tensors provided as a dict and convert them into
    real tensors
    """
    tensor_real = []
    n_max, l_max = get_max(tensor)
    for n in range(n_max):
        for l in range(l_max):
            for m in range(-l,0):
                tensor_real.append((1j/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]-(-1)**m*tensor['{},{},{}'.format(n,l,-m)])).real)
            tensor_real.append(tensor['{},{},{}'.format(n,l,0)].real)
            for m in range(1,l+1):
                tensor_real.append((1/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,-m)]+(-1)**m*tensor['{},{},{}'.format(n,l,m)])).real)

    return np.array(tensor_real)

def make_complex(tensor_array, n_rad, n_l):
    """Take real tensors provided as a np.ndarray and convert them into
    complex tensors represented as a dictionary

    Parameters
    -------
        tensor_array: np.ndarray
            real tensor (ordering: radial ang.momentum projection like: s1 ppp1 ddddd1 s2 etc.)
        n_rad: int,
            number of radials
        n_l: int,
            maximum angular momentum

    Returns
    -------
        dict
            dictionary containing complex tensor elements (keys: {'n,l,m'})
    """
    tensor = {}
    tensor_complex = {}
    cnt = 0
    for n in range(n_rad):
        for l in range(n_l):
            for m in range(-l,l+1):
                tensor['{},{},{}'.format(n,l,m)] = tensor_array[cnt]
                cnt += 1

    for n in range(n_rad):
        for l in range(n_l):
            for m in range(-l,0):
                tensor_complex['{},{},{}'.format(n,l,m)] = ((1/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,-m)]-1j*tensor['{},{},{}'.format(n,l,m)])))
            tensor_complex['{},{},{}'.format(n,l,0)] = (tensor['{},{},{}'.format(n,l,0)]) + 0j
            for m in range(1,l+1):
                tensor_complex['{},{},{}'.format(n,l,m)] = (((-1)**m/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]+1j*tensor['{},{},{}'.format(n,l,-m)])))

    return tensor_complex

def get_casimir(tensor):
    """ Get the casimir element (equiv. to L_2 norm) of a complex tensor

    Parameters
    ---------
        tensor: dict
            dictionary containing tensor in its complex form

    Returns
    -----
        dict
            Casimir element {'n,l'}
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

    Parameters
    -----------
        co: np.ndarray (3,3)
            coordinates of the body-fixed axes in the global coordinate system

    Returns
    -------
        tuple of floats
            euler angles

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

    Parameters
    --------
        vec: np.ndarray (?, 3)
            vector(s) to rotate, note that if more than one vector provided,
            ever vector is rotated by the same angles.

        angles: np.ndarray (3)
            euler angles: alpha, beta, gamma.

        inverse: bool
            {False: rotate vector, True: rotate coordinate system}

    Returns
    -------
        np.ndarray
            rotated vector(s)
    """

    if vec.ndim == 1 and len(vec) == 3:
        vec = vec.reshape(1,3)

    vec = vec[:,[1,2,0]]
    T_inv = np.conj(T.T)

    D = sf.Wigner_D_element(*angles,np.array([1])).reshape(3,3)

    if inverse:
        D = D.conj().T

    # D = D.conj()

    R = T.dot(D.dot(T_inv))
    # assert np.allclose(R.conj(), R)

    vec= np.einsum('ij,kj -> ki', R, vec)

    return vec[:,[2,0,1]].real

def rotate_tensor(tensor, angles, inverse = False):
    """ Rotate a complex tensor.

    Parameters
    ----------
        tensor: dict
            complex rank-2 tensor to rotate; the tensor is expected to be complete
            i.e. no entries should be missing
        angles: np.ndarray (3,)
            euler angles: alpha, beta, gamma
        inverse: bool,
             {False: rotate vector, True: rotate CS}

    Returns
    ---------
        dict
            Rotated version of tensor


    Info
    ----
        Remember that in nncs and elfcs alignment, inverse = True should be used
    """

    if not isinstance(tensor['0,0,0'], np.complex128) and not isinstance(tensor['0,0,0'], np.complex64)\
        and not type(tensor['0,0,0']) == complex:
        raise Exception('tensor has to be complex')
    R = {}

    n_max, l_max = get_max(tensor)
    for l in range(1,l_max):
        # if not '0,{},0'.format(l) in tensor:
            # break
        R[l] = sf.Wigner_D_element(*angles,np.array([l])).reshape(2*l+1,2*l+1)
        if inverse:
            R[l] = R[l].conj().T
        # R[l] = R[l].conj()

    tensor_rotated = {}
    for n in range(n_max):
        # if not '{},0,0'.format(n) in tensor:
            # break

        tensor_rotated['{},0,0'.format(n)] = tensor['{},0,0'.format(n)]
        for l in range(1, l_max):
            # if not '0,{},0'.format(l) in tensor:
                # break
            t = []
            for m in range(-l,l+1):
                t.append(tensor['{},{},{}'.format(n,l,m)])
            t = np.array(t)
            t_rotated = R[l].dot(t)
            for m in range(-l,l+1):
                tensor_rotated['{},{},{}'.format(n,l,m)] = t_rotated[l+m]
    return tensor_rotated

def get_elfcs_angles(i, coords, tensor):
    """ Get angles relating global coordinate system to
        local coordinate system (LCS) defined by electronic structure

        Parameters
        -------
            i: int
                LCS around atom i
            coords: np.ndarray (?, 3)
                all atomic positions in given sytem
            tensor: dict
                complex tensor (electronic descriptor) to use for alignment

        Returns
        -------
            list of floats
                Euler angles alpha, beta, gamma
    """

    # Collect all p-orbitals as vectors
    n_max, l_max = get_max(tensor)
    if l_max > 1:
        p = []
        for n in range(n_max):
            p_real = np.array([tensor['{},1,-1'.format(n)],
                tensor['{},1,0'.format(n)],tensor['{},1,1'.format(n)]])
            p_real = (T.dot(p_real))[[2,0,1]]
            p.append(p_real.real)
        p = np.array(p)

    norm = np.linalg.norm
    len_normal = len(p)
    k = 0
    for k, d in enumerate(p):
        if norm(d) > NORM_THRESHOLD:
            axis1 = p[k]/norm(p[k])
            break
    for u, d in enumerate(p[k:]):
        # Find another p-orbital (or l=1 tensor) that is not collinear
        # with the first axis
        if norm(d) > NORM_THRESHOLD and 1 - abs(np.dot(axis1,d)/(norm(axis1)*norm(d))) > ANGLE_THRESHOLD:
            axis2 = d
            break
    # If everything fails, pick the direction to the nearest atom as
    # the second axis
    else:
        c = np.array(coords[i])
        coords = np.delete(coords, i, axis = 0)
        dr = norm((coords - c), axis =1)
        order = np.argsort(dr)
        for o in order:
            axis2 = coords[o] - c
            if 1 - np.abs(axis2.dot(axis1)/norm(axis2)) > ANGLE_THRESHOLD:
                # print('Axis2 with nn')
                break
        else:
            raise Exception('Could not determine orientation. Aborting...')

    axis3 = np.cross(axis1, axis2)
    axis3 = axis3/norm(axis3)
    axis2 = np.cross(axis3, axis1)
    axis2 = axis2/norm(axis2)
    # Round to avoid problems in arccos of get_euler_angles()
    # 10 digits should be more than enough accuracy given other 'error' sources
    axis1 = axis1.round(10)
    axis2 = axis2.round(10)
    axis3 = axis3.round(10)

    angles = get_euler_angles(np.array([axis1, axis2, axis3]))
    return angles

def get_nncs_angles(i, coords, tensor = None):
    """ Get angles relating global coordinate system to local
        coordinate system (LCS) defined by nearest neighbors

        Parameters
        ---
            i: int
                 LCS around atom i
            coords: np.ndarray (?, 3)
                 all atomic positions in given sytem
            tensor: None
                placeholder

        Returns
        -------
            list of floats
                Euler angles alpha, beta, gamma

    """


    norm = np.linalg.norm
    c = np.array(coords[i])
    coords_sorted = np.array(coords)
    coords_sorted = np.delete(coords_sorted, i , axis = 0)
    order = np.argsort(np.linalg.norm(coords_sorted - c, axis = 1))
    coords_sorted = coords_sorted[order]

    # Direction to nearest atom determines first axis
    axis1 = coords_sorted[0] - c
    axis1 = axis1/norm(axis1)

    # Second axis determined by direction to next nearest atom
    # If collinear with axis1 proceed to next nearest atom
    for u, cs in enumerate(coords_sorted[1:]):
        axis2 = cs - c
        if 1 - np.abs(axis2.dot(axis1)/norm(axis2)) > ANGLE_THRESHOLD:
            break
    else:
        raise Exception('Could not determine orientation. Aborting...')

    axis3 = np.cross(axis1, axis2)
    axis3 /= norm(axis3)
    axis2 = np.cross(axis3, axis1)
    axis2 /= norm(axis2)

    # Round to avoid problems in arccos of get_euler_angles()
    axis1 = axis1.round(10)
    axis2 = axis2.round(10)
    axis3 = axis3.round(10)

    angles = get_euler_angles(np.array([axis1, axis2, axis3]))

    return angles

#TODO: Find faster implementation than recursion, non-ortho implementation

def fold_back_coords(i, coords, unitcell):
    """
    Return the periodic images of coords in a unit-cell
    that are closest to coords[i]

    Parameters
    ------

        i: int
             central atom
        coords: np.ndarray (?, 3)
            all atomic positions in given sytem
        unitcell: np.ndarray (3,3)
            unitcell in angstrom

    Returns
    -------
        np.ndarray (?, 3)
            peridic images of coords
    """

    if not np.allclose(unitcell.astype(bool), np.eye(3).astype(bool)):
        raise Exception('fold_back_coords not implemented for non orthorhombic unitcells')
    else:
        uc = np.diag(unitcell)
    coords = np.array(coords.reshape(-1,3))
    rel_c = coords - coords[i:i+1]
    for u in range(3):
        coords[:,u] -= np.sign(rel_c[:,u])*uc[u]*\
        (np.sign(np.abs(rel_c[:,u]) - uc[u]*.5)+1)*.5

    rel_c = coords - coords[i:i+1]
    if np.all(np.abs(rel_c) < (uc/2).reshape(1,3)):
        return coords
    else:
        return fold_back_coords(i, coords, unitcell)
