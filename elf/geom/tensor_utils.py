import numpy as np
import spherical_functions as sf

def make_real(tensor):
    '''
    Take complex tensors tensor provided as a dict and convert them into
    real tensors
    '''
    tensor_real = []
    for n in range(100):
        if not '{},0,0'.format(n) in tensor:
            break
        for l in range(100):
            if not '{},{},0'.format(n,l) in tensor:
                break
            for m in range(-l,0):
                tensor_real.append((-1j/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,m)]-(-1)**m*tensor['{},{},{}'.format(n,l,-m)])).real)
            tensor_real.append(tensor['{},{},{}'.format(n,l,0)].real)
            for m in range(1,l+1):
                tensor_real.append((-1/np.sqrt(2)*(tensor['{},{},{}'.format(n,l,-m)]+(-1)**m*tensor['{},{},{}'.format(n,l,m)])).real)

    return tensor_real

def get_casimir(tensor):
    ''' Get the casimir element (equiv. to L_2 norm) of a tensor
    '''
    casimir = {}

    for n in range(100):
        if not '{},0,0'.format(n) in tensor:
            break
        for l in range(100):
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



def rotate_tensor(tensor, angles, inverse = False):
    """ Rotate a (complex!!) tensor.

    Parameters:
    ----------
    tensor: dict; rank-2 tensor to rotate; the tensor is expected to be complete
        that is no entries should be missing
    angles: euler angles, {alpha, beta, gamma}
    inverse: boolen; inverse rotation

    Returns:
    ---------
    Rotated version of vec
    """
    if not isinstance(tensor['0,0,0'], np.complex128) and not isinstance(tensor['0,0,0'], np.complex64):
        raise Exception('tensor has to be complex')
    R = {}
    for l in range(1,100):
        if not '0,{},0'.format(l) in tensor:
            break
        R[l] = sf.Wigner_D_element(*angles,np.array([l])).reshape(2*l+1,2*l+1)

    tensor_rotated = {}
    for n in range(100):
        if not '{},0,0'.format(n) in tensor:
            break

        tensor_rotated['{},0,0'.format(n)] = tensor['{},0,0'.format(n)]
        for l in range(1, 100):
            if not '0,{},0'.format(l) in tensor:
                break
            t = []
            for m in range(-l,l+1):
                t.append(tensor['{},{},{}'.format(n,l,m)])
            t = np.array(t)
            if not inverse:
                t_rotated = R[l].T.dot(t)
            else:
                t_rotated = R[l].dot(t)
            for m in range(-l,l+1):
                tensor_rotated['{},{},{}'.format(n,l,m)] = t_rotated[l+m]
    return tensor_rotated

def get_nncs_angles(i, coords):
    """ Get euler angles to rotate to the local CS or coords[i] that is
     oriented according to the nearest neighbor rule
    """
    #TODO: Test for collinearity

    norm = np.linalg.norm
    c = np.array(coords[i])
    coords_sorted = np.array(coords)
    coords_sorted = np.delete(coords_sorted, i , axis = 0)
    order = np.argsort(np.linalg.norm(coords_sorted - c, axis = 1))
    coords_sorted = coords_sorted[order]

    axis1 = coords_sorted[0] - c
    axis1 = axis1/norm(axis1)
    axis2 = coords_sorted[1] - c
    axis3 = np.cross(axis1, axis2)
    axis3 /= norm(axis3)
    axis2 = np.cross(axis3, axis1)
    axis2 /= norm(axis2)

    axis1 = axis1.round(10)
    axis2 = axis2.round(10)
    axis3 = axis3.round(10)

    angles = get_euler_angles(np.array([axis1, axis2, axis3]))

    return angles
