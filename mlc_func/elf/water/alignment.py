from mlc_func.elf.geom import get_euler_angles
import numpy as np

def get_water_angles(i, coords, tensor = None):
    """ Get euler angles to rotate to the water molecule centered CS for
        coords[i]. (Assumes that ordering in coords is OHHOHH...)
    """

    def normalize(vec):
        return vec/np.linalg.norm(vec, axis = -1)

    mol_idx = int(np.floor(i/3))
    mol_coords = coords.reshape(-1,3,3)[mol_idx]

    if i%3 == 2:
        mol_coords[[1,2]] = mol_coords[[2,1]]
    O = mol_coords[0]
    mol_coords -= O
    axis2 = normalize(normalize(mol_coords[1]) + normalize(mol_coords[2]))
    axis3 = normalize(np.cross(axis2,mol_coords[1]))
    axis1 = normalize(np.cross(axis2, axis3))

    # Round to avoid problems in arccos of get_euler_angles()
    axis1 = axis1.round(10)
    axis2 = axis2.round(10)
    axis3 = axis3.round(10)

    angles = get_euler_angles(np.array([axis1, axis2, axis3]))

    return angles
