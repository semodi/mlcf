import numpy as np 
class Density():
    def __init__(self, rho, unitcell, grid):
        if rho.ndim != 3:
            raise Exception('rho.ndim = {}, expected: 3'.format(rho.ndim))
        if unitcell.shape != (3,3):
            raise Exception('unitcell.shape = {}, expected: (3,3)'.format(unitcell.shape))
        if grid.shape != (3):
            raise Exception('grid.shape = {}, expected: (3)'.format(grid.shape))
        self.rho = rho
        self.unitcell = unitcell
        self.grid = grid

    def mesh_3d(rmin=[0, 0, 0], rmax=0, scaled = False, pbc = True, indexing = 'xy'):

        """Returns a 3d mesh taking into account periodic boundary conditions

            Parameters
            ----------
            rmin, rmax: (3) list; lower and upper cutoff
            scaled: boolean; scale the meshes with unitcell size?
            pbc: boolean, assume periodic boundary conditions
            indexing: 'xy' or 'ij'; indexing scheme used by np.meshgrid

            Returns
            -------
            X, Y, Z: np.arrays; meshgrid
        """

        if rmax == 0:
            mid_grid = np.floor(self.grid / 2).astype(int)
            rmax = mid_grid

        # resolve the periodic boundary conditions
        if pbc:
            x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0]+1))
            y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1]+1))
            z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2]+1))
        else:
            x_pbc = list(range(rmin[0], rmax[0] +1 )) + list(range(-rmax[0], -rmin[0]))
            y_pbc = list(range(rmin[1], rmax[1] +1 )) + list(range(-rmax[1], -rmin[1]))
            z_pbc = list(range(rmin[2], rmax[2] +1 )) + list(range(-rmax[2], -rmin[2]))


        Xm, Ym, Zm = np.meshgrid(x_pbc, y_pbc, z_pbc, indexing = indexing)

        U = np.array(self.unitcell) # Matrix to go from real space to mesh coordinates
        for i in range(3):
            U[i,:] = U[i,:] / self.grid[i]

        a = np.linalg.norm(self.unitcell, axis = 1)/self.grid[:3]

        Rm = np.concatenate([Xm.reshape(*Xm.shape,1),
                             Ym.reshape(*Xm.shape,1),
                             Zm.reshape(*Xm.shape,1)], axis = 3)
