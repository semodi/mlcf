from .network import Network, Subnet, Dataset
import mlc_func.elf as elf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import keras
from sklearn.preprocessing import MinMaxScaler, Normalizer,StandardScaler
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import os
import pickle

class Force_Network():
    """ MLCF for force perdiction"""
    def __init__(self, species, scaler, basis, datasets = {}, mask = [], n_layers = 3, nodes_per_layer = 8,
                b = 0):
        """
        Parameters
        ----------

        species: str, chemical element symbol
        scaler: sklearn Scaler
        basis: dict, basis that was used to create electronic descriptors
        datasets: dict, datasets provided as {'X_train': np.ndarray, 'X_test': etc...}
        mask: list of bool, used to mask the features and filter out features with low variance
        n_layers: int, number of hidden layers, default = 3
        nodes_per_layer: int, nodes for each hidden layer, default = 8
        b: float, l2-regularization strenght, default = 0
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

        Parameters:
        -----------
        step_size: float, step size to take during gradient descent, default=0.001
        max_epochs: int, max. number of epochs to train, default=50001
        b: float, l2-regularization
        early_stopping: bool, use early stopping (interrupt training once valid loss increases),
            default=False
        batch_size: int, number of samples per batch, default=500
        epochs_per_output: int, only print overview every epochs_per_output steps, default=500
        restart: bool, restart training from beginning (reset network), default=False
        tol_train: float, stop training if relative value of training loss decreases by less than this value
        tol_valid: float, stop training if relative value of validation loss decreases by less than this value
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

        Parameters:
        ----------
        feat: np.ndarray, input features
        processed: bool, are features processed (scaled, masked)?

        Returns:
        -------
        np.ndarray, predicted forces
        """
        if not processed:
            return self.model.predict(self.scaler.transform(feat[:,self.mask]))
        else:
            return self.model.predict(feat)

    def evaluate(self, plot = False, on = 'test'):
        """ Evaluate model performance

        Parameters:
        -----------
        plot: bool, plot correlation plots
        on: {'test','train','valid'} which set to evaluat on

        Returns:
        --------
        dict, containing rmse, mae and max. abs. error
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

        Parameters:
        -----------
        net_dir: str, directory to save mlcf to
        override: bool, if net_dir already contains model, allow to override?
            default = False
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

        Parameters:
        ----------
        net_dir: str, path to directory containing MLCF
        """
        supp = pickle.load(open(net_dir + 'supp_' + self.species, 'rb'))
        self.model = keras.models.load_model(net_dir + 'force_' + self.species)
        self.mask = supp['mask']
        self.scaler = supp['scaler']
        self.basis = supp['basis']
        self.compiled = True

    def learning_curve(self, steps = 5):
        """Create a learning curve by varying the training set size

        Parameters:
        -----------
        steps: int, how many different training set sizes to use

        Returns:
        ---------

        dict = {'N': training set size,'train': training loss,'valid': validation loss}
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

        Parameters:
        ----------
        path:  path to .hdf5 file containing features

        Returns:
        -------
        np.ndarray, force prediction

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

    Parameters:
    ----------
    net_dir: str, path to directory containing MLCF
    species: str, specifies which chemical element to load model for
    """
    
    model = Force_Network(species, None, None, mask = [True])
    model.load_all(net_dir)
    return model
