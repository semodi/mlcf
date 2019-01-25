import mlc_func as mlcf
import numpy as np
import pandas as pd
from ase.io import read, write
import pickle
import pandas as pd
from ase import Atoms
import time
from sklearn.gaussian_process.kernels import RBF
import pipelines

energy_model = mlcf.ml.load_energy_model('./simple_monomer/')
bmodels = pickle.load(open('./monomer_dmodel', 'rb'))

def dDescr(i, all_pos, species, bmodels, masks, dx = 0.0001):

    dDescr_list = []
    pipeline = {}
    target_scaler = {}
    for pos in all_pos: #Loop over systems
        sys_dDescr = []
        for j,(p, spec) in enumerate(zip(pos, species)): # Loop over atoms in system
            if not spec in pipeline:
                pipeline[spec], target_scaler[spec] = pickle.loads(bmodels[spec.lower()]).values()
            X = [{'pos': pos, 'angle' : mlcf.elf.geom.get_nncs_angles(j, pos, None)}]

            # How does rs desciptor around atom j change if I move atom i
            dx = dX(X, i, j ,pipeline[spec])

            # How does electron density descriptor around atom j change with rs around atom j
            g = G(X, pipeline[spec],target_scaler[spec],masks[spec.lower()], j)


            sys_dDescr.append(g.dot(dx.T))
        dDescr_list.append(np.concatenate(sys_dDescr, axis = 0))

    return np.array(dDescr_list)

def dX(X, i, j, pipeline, dx = 0.0001):
    """ Gradient in the input features (the transformed coordinates)
    moving atom: i
    real space descriptors around atom j
    """
    preprocessor = pipeline.steps[0][1]
    preprocessor.u = j
    pos0 = X[0]['pos']
    angle = X[0]['angle']
    X_new = []
    for k in range(3):
        dr = np.zeros_like(pos0)
        dr[i,k] = dx
        X_new.append({'pos': pos0 + dr, 'angle' : mlcf.elf.geom.get_nncs_angles(j, pos0 + dr, None)})
#         X_new.append({'pos': pos0 + dr, 'angle' : angle})
    for k in range(3):
        dr = np.zeros_like(pos0)
        dr[i,k] = -dx
        X_new.append({'pos': pos0 + dr, 'angle' : mlcf.elf.geom.get_nncs_angles(j, pos0 + dr, None)})
#         X_new.append({'pos': pos0 + dr, 'angle' : angle})
    X_new_transformed = preprocessor.transform(X_new)
    X_new_transformed = X_new_transformed.reshape(2,-1)
    return ((X_new_transformed[0] - X_new_transformed[1])/(2*dx)).reshape(3,-1)


def G(X, pipeline, target_scaler, mask, j):
    """ Gradient of the regressor (KRR)
        Change in descriptors associated with atom j
    """

    krr = pipeline.steps[1][1]
    preprocessor = pipeline.steps[0][1]
    preprocessor.u = j
    X_transformed = preprocessor.transform(X)

    rbf = RBF(length_scale=np.sqrt(.5/krr.gamma))
    result = -(krr.dual_coef_.T.dot((X_transformed - krr.X_fit_)*rbf(krr.X_fit_, X_transformed))*2*krr.gamma)
    return target_scaler.inverse_transform(result.T, std_only = True)[:,mask].T
#     return result


def get_energy(descr):
    energy = 0

    for spec in ['O','H']:
        shape = descr[spec].shape
        energy += np.sum(energy_model.predict(descr[spec].reshape(-1,shape[-1]), spec,
                                              use_masks=True))
    return energy

def get_forces(descr, atoms):
    print('Correcting Forces')
    gradients = []
    for spec in ['O','H']:
        shape = descr[spec].shape
        gradients.append(energy_model.predict(descr[spec].reshape(-1,shape[-1]), spec,
                                              use_masks=True,
                                              return_gradient=True)[1].reshape(shape[0],-1))
    gradients = np.concatenate(gradients, axis = 1)
    gradients = gradients.reshape(len(gradients), -1)
    species = atoms[0].get_chemical_symbols()
    dd = np.concatenate([dDescr(i,[a.get_positions() for a in atoms],
                                species,
                                bmodels, energy_model.masks) for i in range(len(species))],
                        axis = -1)
    predictions = -np.einsum('ijk, ij -> ik', dd, gradients).reshape(-1,3,3)
    return predictions
