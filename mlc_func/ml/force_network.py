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

    def __init__(self, species, scaler, basis, datasets = {}, mask = [], n_layers = 3, nodes_per_layer = 8,
                b = 0):


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
        self.scaler = scaler
        self.basis = basis
        if len(mask) == 0:
            mask = [True]*self.X_train.shape[1]
        else:
            self.mask = mask

        if not (scaler == None or basis == None):
            self.build_model()


    def build_model(self, override = False):
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
              restart = False):


        if not self.compiled or restart:
            self.learning_rate = step_size
            self.b = b
            self.build_model(override = True)
            self.__compile_model(override=True)
        elif step_size != self.learning_rate or b != self.b:
            if not restart:
                raise Exception('Step size and regularization can not be changed after training was started, please set restart = True')

        last_train = 1e8
        last_valid = 1e8
        for i in range(int(max_epochs/epochs_per_output)):
            self.model.fit(self.X_train, self.y_train, epochs=epochs_per_output,
             batch_size=batch_size, verbose=0)
            train_loss = np.sqrt(self.model.evaluate(self.X_train, self.y_train, verbose = 0)[0])
            valid_loss = np.sqrt(self.model.evaluate(self.X_valid, self.y_valid, verbose = 0)[0])
            if train_loss < last_train and valid_loss > last_valid and early_stopping:
                break  #Early stopping
            else:
                last_train = train_loss
                last_valid = valid_loss
            print('--------Epoch = {}----------'.format(i*epochs_per_output))
            print('Training loss || Validation loss')
            print( '{:13.6f} || {:13.6f}'.format(train_loss, valid_loss))


    def predict(self, feat, processed = False):
        if not processed:
            return self.model.predict(self.scaler.transform(feat[:,self.mask]))
        else:
            return self.model.predict(feat)

    def evaluate(self, plot = False, on = 'test'):
        X, y = {'train':[self.X_train, self.y_train],
                'valid':[self.X_valid, self.y_valid],
                'test':[self.X_test, self.y_test]}[on]

        prediction = self.predict(X, True)
        error = y - prediction
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max = np.max(np.abs(error))

        print('======== Evaluation on test set =============\n\
              RMSE =  {:5.4f}\n\
              MAE = {:5.4f}\n\
              Max. abs. error = {:5.4f}'.format(rmse, mae, max))
        if plot:
            plt.figure()
            plt.plot(y, prediction, ls = '', marker = '.')
            plt.xlabel('Expected')
            plt.ylabel('Predicted')
            plt.show()

        return {'rmse' : rmse, 'mae': mae, 'max': max}

    def save_all(self, net_dir, override = False):
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
        supp = pickle.load(open(net_dir + 'supp_' + self.species, 'rb'))
        self.model = keras.models.load_model(net_dir + 'force_' + self.species)
        self.mask = supp['mask']
        self.scaler = supp['scaler']
        self.basis = supp['basis']
        self.compiled = True

def load_force_model(net_dir, species):
    model = Force_Network(species, None, None, mask = [True])
    model.load_all(net_dir)
    return model
