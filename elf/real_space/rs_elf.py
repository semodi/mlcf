import numpy as np
from scipy.special import sph_harm
import scipy.linalg
from sympy.physics.wigner import clebsch_gordan as CG
from sympy import N
from functools import reduce
import time
from ase import Atoms
from .density import Density
from ase.units import Bohr
from elf.geom import get_nncs_angles, get_elfcs_angles
from elf.geom import make_real, rotate_tensor, fold_back_coords
from elf import ElF
from elf.utils import serial_view


def box_around(pos, radius, density, unit = 'A'):
    '''
    Return dictionary containing box around an atom at position pos with
    given radius. Dictionary contains box in mesh, euclidean and spherical
    coordinates
    '''

    if pos.shape != (1,3) and (pos.ndim != 1 or len(pos) !=3):
        raise Exception('please provide only one point for pos')

    pos = pos.flatten()

    U = np.array(density.unitcell) # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i,:] = U[i,:] / density.grid[i]
    a = np.linalg.norm(density.unitcell, axis = 1)/density.grid[:3]
    U = U.T

    #Create box with max. distance = radius
    rmax = np.ceil(radius / a).astype(int).tolist()
    Xm, Ym, Zm = density.mesh_3d(scaled = False, pbc= False, rmax = rmax, indexing = 'ij')
    X, Y, Z = density.mesh_3d(scaled = True, pbc= False, rmax = rmax, indexing = 'ij')

    #Find mesh pos.
    cm = np.round(np.linalg.inv(U).dot(pos)).astype(int)
    dr = pos  - U.dot(cm)
    X -= dr[0]
    Y -= dr[1]
    Z -= dr[2]

    Xm = (Xm + cm[0])%density.grid[0]
    Ym = (Ym + cm[1])%density.grid[1]
    Zm = (Zm + cm[2])%density.grid[2]

    R = np.sqrt(X**2 + Y**2 + Z**2)

    Phi = np.arctan2(Y,X)
    Theta = np.arccos(Z/R, where = (R != 0))
    Theta[R == 0] = 0

    return {'mesh':[Xm, Ym, Zm],'real': [X,Y,Z],'radial':[R, Theta, Phi]}

def g(r, r_i, r_c, a, gamma):
    """
    Non-orthogonalized radial functions
    """
    def g_(r, r_i, r_c, a):
        return (r-r_i)**(2)*(r_c-r)**(a+2)*np.exp(-gamma*(r/r_c)**(1/4))
#          return (r-r_i)**(5)*(r_c-r)**(a+2)
    r_grid = np.arange(r_i, r_c, (r_c-r_i)/1e3)
    N = np.sqrt(np.sum(g_(r_grid,r_i,r_c, a)*g_(r_grid,r_i,r_c,a))*(r_c-r_i)/1e3)
    return g_(r,r_i,r_c,a)/N

def S(r_i, r_o, nmax, gamma):
    '''
    Overlap matrix between radial basis functions
    '''

    S_matrix = np.zeros([nmax,nmax])
    r_grid = np.arange(r_i, r_o, (r_o-r_i)/1e3)
    for i in range(nmax):
        for j in range(i,nmax):
            S_matrix[i,j] = np.sum(g(r_grid,r_i,r_o,i+1, gamma)*g(r_grid,r_i,r_o,j+1, gamma))*(r_o-r_i)/1e3
    for i in range(nmax):
        for j in range(i+1, nmax):
            S_matrix[j,i] = S_matrix[i,j]
    return S_matrix


def radials(r, r_i, r_o, W, gamma):
    '''
    Get orthonormal radial basis functions
    '''
    result = np.zeros([len(W)] + list(r.shape))
    for k in range(0,len(W)):
        rad = g(r,r_i,r_o,k+1, gamma)
        for j in range(0, len(W)):
            result[j] += W[j,k] * rad
    result[:,r > r_o] = 0
    result[:,r < r_i] = 0
#    result = result[::-1] # Invert so that n = 0 is closest to origin
    return result

def get_W(r_i, r_o, n, gamma):
    '''
    Get matrix to orthonormalize radial basis functions
    '''
    return scipy.linalg.sqrtm(np.linalg.pinv(S(r_i,r_o, n, gamma)))

def decompose(rho, box, n_rad, n_l, r_i, r_o, gamma, V_cell = 1):
    '''
    Parameters:
    ----------
    rho: np.ndarray; electron charge density on grid
    box: dict; contains the mesh in spherical and euclidean coordinates
    n_rad: int; number of radial functions
    n_l: int; number of spherical harmonics
    r_i: float; inner radial cutoff in Angstrom
    r_o: float; outer radial cutoff in Angstrom
    gamma: float; exponential damping
    V_cell: float; volume of one grid cell

    Returns:
    --------
    dict; dictionary containing the complex ELF
    '''

    R, Theta, Phi = box['radial']
    Xm, Ym, Zm = box['mesh']

    # Automatically detect whether entire charge density or only surrounding
    # box was provided
    if rho.shape == Xm.shape:
        small_rho = True
    else:
        small_rho = False

    #Build angular part of basis functions
    angs = []
    for l in range(n_l):
        angs.append([])
        for m in range(-l,l+1):
            # angs[l].append(sph_harm(m, l, Phi, Theta).conj()) TODO: In theory should be conj!?
            angs[l].append(sph_harm(m, l, Phi, Theta))

    #Build radial part of b.f.
    W = get_W(r_i, r_o, n_rad, gamma) # Matrix to orthogonalize radial basis
    rads = radials(R, r_i, r_o, W, gamma)

    coeff = {}
    if small_rho:
        for n in range(n_rad):
            for l in range(n_l):
                for m in range(2*l+1):
                    coeff['{},{},{}'.format(n,l,m-l)] = np.sum(angs[l][m]*rads[n]*rho)*V_cell
    else:
        for n in range(n_rad):
            for l in range(n_l):
                for m in range(2*l+1):
                    coeff['{},{},{}'.format(n,l,m-l)] = np.sum(angs[l][m]*rads[n]*rho[Xm, Ym, Zm])*V_cell
    return coeff

def atomic_elf(pos, density, basis, chem_symbol):
    '''
    Given an input density and an atomic position decompose the
    surrounding charge density into an ELF

    Parameter:
    ----------
    pos: (,3) np.ndarray; atomic position
    density: density object; stores charge density rho, unitcell, and grid
                (see density.py)
    basis: dict; specifies the basis set used for the ELF decomposition
                        for each chem. element
    chem_symbol: str; chemical symbol

    Returns:
    --------
    dict; dictionary containing the real ELF '''

    chem_symbol = chem_symbol.lower()

    if pos.shape == (3,):
        pos = pos.reshape(1,3)
    if pos.shape != (1,3):
        raise Exception('pos has wrong shape')

    U = np.array(density.unitcell) # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i,:] = U[i,:] / density.grid[i]
    V_cell = np.linalg.det(U)

    # The following two lines are needed to
    #obtain the dataset from the old implementation
    V_cell /= (37.7945/216)**3*Bohr**3
    V_cell *= np.sqrt(Bohr)

    box = box_around(pos, basis['r_o_' + chem_symbol], density)
    coeff = decompose(density.rho, box,
                           basis['n_rad_' + chem_symbol],
                           basis['n_l_' + chem_symbol],
                           basis['r_i_' + chem_symbol],
                           basis['r_o_' + chem_symbol],
                           basis['gamma_' + chem_symbol],
                           V_cell = V_cell)

    return coeff

def get_elf_oriented_thread(pos, density, basis, chem_symbol,
    i, all_positions, mode):
    """ Method that should be used in a parallel executions.
    One thread/process computes and orients the ElF for a single atom
    inside a system
    """
    e = ElF(atomic_elf(pos, density, basis, chem_symbol),[0,0,0],basis,
        chem_symbol,density.unitcell)
    elf_oriented = orient_elf(i,e,all_positions,mode)

    return(elf_oriented)


def get_elfs(atoms, density, basis, view = serial_view(), orient_mode = 'none'):
    '''
    Given an input density and an ASE Atoms object decompose the
    complete charge density into atomic ELFs

    Parameter:
    ----------
    atoms: ASE atoms instance
    density: density instance; stores charge density rho, unitcell, and grid
                (see density.py)
    basis: dict; specifies the basis set used for the ELF decomposition
                        for each chem. element

    view: ipyparallel balanced view for parallel execution through sync map
    orient_mode = {'none': do not orient and return complex tensor,
                   'elf'/'nn': orient using the elf or nn algorithm and return
                   real tensor}
    Returns:
    --------
    list; list containing the complex/real atomic ELFs '''

    density_list = []
    pos_list = []
    sym_list = []
    basis_list = []

    for pos, sym in zip(atoms.get_positions(), atoms.get_chemical_symbols()):
        rel_basis = {} #relevant basis entries
        for b in basis:
            if sym.lower() == b.lower()[-1]:
                rel_basis[b] = basis[b]
        if len(rel_basis) == 0: continue   # Skip atoms for which no basis provided

        box = box_around(pos, basis['r_o_' + sym.lower()], density)
        density_list.append(Density(density.rho[box['mesh']],
                                                density.unitcell,
                                                density.grid))
        pos_list.append(pos)
        sym_list.append(sym)
        basis_list.append(rel_basis)

    if orient_mode == 'none':
        values = view.map_sync(atomic_elf, pos_list, density_list, basis_list, sym_list)
        elfs = []
        for v,b,s in zip(values, basis_list, sym_list):
            elfs.append(ElF(v,[0,0,0],b, s, density.unitcell))
    else:
        n_jobs = len(basis_list)
        all_pos = atoms.get_positions()
        elfs = view.map_sync(get_elf_oriented_thread, pos_list, density_list,
          basis_list, sym_list, list(range(n_jobs)),[all_pos]*n_jobs,
          [orient_mode]*n_jobs)

    return elfs

def get_elfs_oriented(atoms, density, basis, mode, view = serial_view()):
    """
    Outdated, use get_elfs() with "mode='elf'/'nn'" instead.

    Like get_elfs, but returns real, oriented elfs
    mode = {'elf': Use the ElF algorithm to orient fingerprint,
            'nn': Use nearest neighbor algorithm}
    """
    return get_elfs(atoms, density, basis, view, orient_mode = mode)

def orient_elf(i, elf, all_pos, mode):
    """
    Takes an ElF and orient it according
    to the rule specified in mode.

    Parameters:
    -----------
    i: int; Index of the atom in all_pos
    elf: ElF; ElF to orient
    all_pos: numpy.ndarray; positions of all atoms in system (including the
    one with index i)
    mode = {'elf': Use the ElF algorithm to orient fingerprint,
                'nn': Use nearest neighbor algorithm}
    """
    oriented_elfs = []
    if mode == 'elf':
        angles_getter = get_elfcs_angles
    elif mode == 'nn':
        angles_getter = get_nncs_angles
    else:
        raise Exception('Unkown mode: {}'.format(mode))

    angles = angles_getter(i, fold_back_coords(i, all_pos, elf.unitcell), elf.value)
    oriented = make_real(rotate_tensor(elf.value, np.array(angles), True))
    elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell)
    return elf_oriented

def orient_elfs(elfs, atoms, mode):
    """Convenience function that applies orient_elf to a list of elfs.
       (Exists for compatibility reasons)
    """

    oriented_elfs = []
    for i, elf in enumerate(elfs):
        oriented_elfs.append(orient_elf(i ,elf, atoms.get_positions(),mode))

    return oriented_elfs
