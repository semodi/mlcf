""" Module that implements Force_Network, the machine learned correcting functional (MLCF)
for forces
"""

import mlc_func.elf as elf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pickle
import h5py
import json
from ase.io import read

class Force_Network():
    def __init__(self, species, scaler, basis, datasets = {}, mask = [], n_layers = 3, nodes_per_layer = 8,
                b = 0):
        """ MLCF for force perdiction

        Parameters
        ----------

            species: str
                chemical element symbol
            scaler: sklearn Scaler
            basis: dict
                basis that was used to create electronic descriptors
            datasets: dict
                datasets provided as {'X_train': np.ndarray, 'X_test': etc...}
            mask: list of bool
                used to mask the features and filter out features with low variance
            n_layers: int
                number of hidden layers, default = 3
            nodes_per_layer: int
                nodes for each hidden layer, default = 8
            b: float
                l2-regularization strenght, default = 0
        """

        self.species = species
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.datasets = datasets
        self.compiled = False
        self.learning_rate = 0.001
        self.model = None
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9,
            beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.b = b
        if not len(datasets) == 0:
            self.X_train = datasets['X_train']
            self.X_test = datasets['X_test']
            self.X_valid = datasets['X_valid']
            self.y_train = datasets['y_train']
            self.y_test = datasets['y_test']
            self.y_valid = datasets['y_valid']
        else:
            self.X_train = None
        self.scaler = scaler
        self.basis = basis
        if len(mask) == 0:
            mask = [True]*self.X_train.shape[1]
        else:
            self.mask = mask

        if not (scaler == None or basis == None):
            self.__build_model()


    def __build_model(self, override = False):
        if self.compiled and not override:
            raise Exception('Model already compiled! Set override = True to proceed.')

        s = self.nodes_per_layer
        self.model = Sequential()
        self.model.add(Dense(units=s, activation='sigmoid',
         kernel_regularizer=regularizers.l2(self.b), input_dim=self.X_train.shape[1]))

        for _ in range(self.n_layers-1):
            self.model.add(Dense(units=s, activation='sigmoid', kernel_regularizer=regularizers.l2(self.b)))
        self.model.add(Dense(units=3, activation='linear'))

    def __compile_model(self, override = False):
        if self.compiled and not override:
            raise Exception('Model already compiled! Set override = True to proceed.')
        else:
            self.model.compile(loss='mean_squared_error',
              optimizer=self.optimizer,
              metrics=['accuracy'])

        self.compiled = True

    def train(self,
              step_size=0.001,
              max_epochs=50001,
              b=0,
              early_stopping = False,
              batch_size = 500,
              epochs_per_output = 500,
              restart = False,
              tol_train = 0,
              tol_valid = 0):
        """ Train the model

        Parameters
        -----------
            step_size: float
                step size to take during gradient descent, default=0.001
            max_epochs: int
                max. number of epochs to train, default=50001
            b: float
                l2-regularization
            early_stopping: bool
                use early stopping (interrupt training once valid loss increases),
                default=False
            batch_size: int
                number of samples per batch, default=500
            epochs_per_output: int
                only print overview every epochs_per_output steps, default=500
            restart: bool
                restart training from beginning (reset network), default=False
            tol_train: float
                stop training if relative value of training loss decreases by less than this value
            tol_valid: float
                stop training if relative value of validation loss decreases by less than this value
        Returns
        ----

            None
        """

        if not self.compiled or restart:
            self.learning_rate = step_size
            self.b = b
            self.__build_model(override = True)
            self.__compile_model(override=True)
        elif step_size != self.learning_rate or b != self.b:
            if not restart:
                raise Exception('Step size and regularization can not be changed after training was started, please set restart = True')

        last_train = 1e8
        last_valid = 1e8
        for i in range(int(max_epochs/epochs_per_output)):

            train_loss = np.sqrt(self.model.evaluate(self.X_train, self.y_train, verbose = 0)[0])
            valid_loss = np.sqrt(self.model.evaluate(self.X_valid, self.y_valid, verbose = 0)[0])
            if train_loss < last_train and valid_loss > last_valid and early_stopping:
                break  #Early stopping
            elif ((last_train - train_loss)/last_train) < tol_train:
                return 0
            elif ((last_valid - valid_loss)/last_valid) < tol_valid:
                return 0
            else:
                last_train = train_loss
                last_valid = valid_loss

            print('--------Epoch = {}----------'.format(i*epochs_per_output))
            print('Training loss || Validation loss')
            print( '{:13.6f} || {:13.6f}'.format(train_loss, valid_loss))
            self.model.fit(self.X_train, self.y_train, epochs=epochs_per_output,
             batch_size=batch_size, verbose=0)


    def predict(self, feat, processed = False):
        """ Get predicted forces

        Parameters
        ----------
            feat: np.ndarray
                input features
            processed: bool
                are features processed (scaled, masked)?

        Returns
        -------
            np.ndarray
                predicted forces
        """
        if not processed:
            return self.model.predict(self.scaler.transform(feat[:,self.mask]))
        else:
            return self.model.predict(feat)

    def evaluate(self, plot = False, on = 'test'):
        """ Evaluate model performance

        Parameters
        --------
            plot: bool
                plot correlation plots
            on: str
                {'test','train','valid'} which set to evaluat on

        Returns
        --------
            dict
                containing rmse, mae and max. abs. error
        """
        X, y = {'train':[self.X_train, self.y_train],
                'valid':[self.X_valid, self.y_valid],
                'test':[self.X_test, self.y_test]}[on]

        prediction = self.predict(X, True)
        error = y - prediction
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max = np.max(np.abs(error))

        print('======== Evaluation on ' + on +' set =============\n\
              RMSE =  {:5.4f}\n\
              MAE = {:5.4f}\n\
              Max. abs. error = {:5.4f}'.format(rmse, mae, max))
        if plot:
            plt.figure()
            for i, l in enumerate(['x','y','z']):
                plt.plot(y[:,i], prediction[:,i], ls = '', marker = '.',
                 label = l)
            plt.xlabel('Expected')
            plt.ylabel('Predicted')
            plt.legend()
            plt.show()

        return {'rmse' : rmse, 'mae': mae, 'max': max}

    def save_all(self, net_dir, override = False):
        """ Save force MLCF

        Parameters
        -----------
            net_dir: str
                directory to save mlcf to
            override: bool
                if net_dir already contains model, allow to override?
                default = False

        Returns
        -------

            None
        """
        if net_dir[-1] != '/': net_dir += '/'
        to_save = {'mask': self.mask, 'scaler': self.scaler,
                   'basis': self.basis}

        if not os.path.exists(net_dir):
            os.mkdir(net_dir)

        if os.path.exists(net_dir + 'force_' + self.species) and not override:
            raise Exception('Already exists, to proceed set override = True')
        else:
            pickle.dump(to_save, open(net_dir + 'supp_' + self.species, 'wb'))
            self.model.save(net_dir + 'force_' + self.species)

    def load_all(self, net_dir):
        """ Load force MLCF from net_dir

        Parameters
        ----------
            net_dir: str
                path to directory containing MLCF
        """
        supp = pickle.load(open(net_dir + 'supp_' + self.species, 'rb'))
        self.model = keras.models.load_model(net_dir + 'force_' + self.species)
        self.mask = supp['mask']
        self.scaler = supp['scaler']
        self.basis = supp['basis']
        self.compiled = True

    def learning_curve(self, steps = 5):
        """Create a learning curve by varying the training set size

        Parameters
        -----------
            steps: int
                how many different training set sizes to use

        Returns
        ---------

            dict,
                {'N': training set size,'train': training loss,
                'valid': validation loss}
        """
        tot_len = len(self.X_train)
        save_X, save_y = np.array(self.X_train), np.array(self.y_train)
        chunk_size = np.floor(tot_len/steps).astype(int)
        N = []
        train_rmse = []
        valid_rmse = []
        for s in range(steps+1):
            train_size = int(tot_len/(2**steps)*2**s)
            N.append(train_size)
            self.X_train = save_X[:train_size]
            self.y_train = save_y[:train_size]
            print('Size = {}'.format(len(self.X_train)))
            self.train(batch_size = np.ceil((len(self.X_train)/100)).astype(int),
                        epochs_per_output = 100,
                        tol_train = 1e-2,
                        tol_valid = 1e-2)
            train_rmse.append(self.evaluate(on='train')['rmse'])
            valid_rmse.append(self.evaluate(on='valid')['rmse'])
        self.X_train = save_X
        self.y_train = save_y
        return {'N': N, 'train': train_rmse, 'valid': valid_rmse}

    def predict_from_hdf5(self, path):
        """ Get force prediction but instead of providing features,
        give source path where features are found

        Parameters
        ----------
            path: str
                path to .hdf5 file containing features

        Returns:
        -------
            np.ndarray
                 force prediction

        """
        elfs, angles = elf.utils.hdf5_to_elfs_fast(path)
        if self.species in elfs:
            species = self.species
            n_samples = len(elfs[species])
            elfs[species] = elfs[species].reshape(-1,elfs[species].shape[-1])
            angles[species] = angles[species].reshape(-1,3)
            predictions = self.predict(elfs[species], False)
            for i, (value, a) in enumerate(zip(predictions, angles[species])):
                predictions[i] = elf.geom.rotate_vector(np.array([value]),
                                                                a, False)
        return predictions.reshape(n_samples,-1,3)


def load_force_model(net_dir, species):
    """ Load force MLCF from net_dir for a given element

    Parameters
    ----------

        net_dir: str
            path to directory containing MLCF
        species: str
            specifies which chemical element to load model for
    """

    model = Force_Network(species, None, None, mask = [True])
    model.load_all(net_dir)
    return model

def build_force_mlcf(feature_src, target_src, traj_src, species, mask = [], filters = [],
    automask_std = 0, autofilt_percent = 0, test_size = 0.2,
    random_state = 42):
    ''' Return a trainable force MLCF (neural network)

    Parameters
    ----------

        feature_src: list
            list of paths to the hdf5 containing the features
        target_src: list
            list of paths to the csv files containing the target forces
            entries in target_scr and feature_src correspond to each other
        traj_src: list
            list of paths to the .traj/.xyz files (needed to determine species
            of each atom)
        species: string
            containing the species that model should be fitted for
        mask: list
            containing booleans; can be used to select which features to use.
            default: use all features
        filters: list
            containing list of booleans; can be used to exclude datapoints
            in sets (e.g. outliers)
        automask_std: float
            if mask not set exclude all features whose stdev across dataset
            is smaller than this value
        autofilt_percent: float
            exclude this percentile of extreme datapoints from set
            (only if filters not set)
        test_size: float
            relative size of hold_out (test) set
        random_state: int
            state used to perform shuffle before spliting dataset

    Returns
    -------

    Force_Network
    '''

    species = species.lower()
    if not len(species) == 1:
        raise Exception('Please specify only one species.')
    all_targets = []
    all_features = []

    if len(filters) != len(feature_src):
        filters = [0]*len(feature_src)

    basis = {}

    for fsrc, tsrc, trsrc, filter in zip(feature_src, target_src, traj_src, filters):
        # elfs = np.array(elf.utils.hdf5_to_elfs(fsrc,
        #                       grouped = True, values_only = True)[species])
        # angles = np.array(elf.utils.hdf5_to_elfs(fsrc,
        #                       grouped = True, angles_only = True)[species])
        elfs, angles = elf.utils.hdf5_to_elfs_fast(fsrc, species)
        elfs = elfs[species]
        angles = angles[species]
        with h5py.File(fsrc) as file:
            this_basis = json.loads(file.attrs['basis'])
            # Filter for species
            species_basis = {}
            for entry in this_basis:
                if entry[-1] == species or\
                 entry == 'alignment':
                 species_basis[entry] = this_basis[entry]

            if len(basis) > 0 and species_basis != basis:
                raise Exception('Basis used across datasets not consistent')
            else:
                basis = species_basis

        angles = angles.reshape(-1,3)

        elfs = elfs.reshape(-1,elfs.shape[-1])
        targets = np.genfromtxt(tsrc, delimiter = ',')
        if not trsrc.split('.')[-1] in ['xyz', 'traj']:
            raise Exception('Invalid file format for trajectory file stored at {}'.format(trsrc))
        traj = read(trsrc, ':')

        all_symbols = np.array([t.get_chemical_symbols() for t in traj]).flatten()
        targets = targets[all_symbols == species.upper()]

        print(elfs.shape)
        for idx, (t, ang) in enumerate(zip(targets, angles)):
            targets[idx] = elf.geom.rotate_vector(np.array([t]),ang.tolist(), inverse=True)

        if not len(elfs) == len(targets):
            raise Exception('Sample sizes inconsistent.')

        if not isinstance(filter, list) and not isinstance(filter, np.ndarray):
            percentile_cutoff = autofilt_percent
            selection = []
            for t in targets.T:
                lim1 = np.percentile(t, percentile_cutoff*100)
                lim2 = np.percentile(t, (1 - percentile_cutoff)*100)
                min_lim, max_lim = min(lim1,lim2), max(lim1,lim2)
                selection.append((t > min_lim) & (t < max_lim))

            filter = [s1 & s2 & s3 for s1,s2,s3 in zip(*selection)]

        if len(mask) != elfs.shape[-1]:
            feat = np.array(elfs)
            mask = (np.std(feat.reshape(-1,feat.shape[-1]),
                        axis = 0) > automask_std)
        all_features.append(elfs[:,mask][filter])
        all_targets.append(targets[filter])

    feat = np.concatenate(all_features)
    targets = np.concatenate(all_targets)

    scaler = StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    X_train, X_test, y_train, y_test = train_test_split(feat,
                                                    targets,
                                                    shuffle =True,
                                                    random_state = random_state,
                                                    test_size = test_size)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                          y_train,
                                                          shuffle =True,
                                                          random_state = random_state,
                                                          test_size = 0.2)
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_test': y_test,
        'y_valid': y_valid
    }
    return Force_Network(species, scaler, basis, datasets, mask)
