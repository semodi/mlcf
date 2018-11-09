from .force_network import Force_Network, load_force_model
import numpy as np
import copy

class Ensemble_Network():

    def __init__(self, network, n = 3):
        self.models = []
        for _ in range(n):
            self.models.append(copy.copy(network))

        if isinstance(network.X_train, np.ndarray):
            X_train, y_train, self.X_test, self.y_test =\
                network.X_train, network.y_train, network.X_test, network.y_test
            self.X_valid = network.X_valid
            self.y_valid = network.y_valid

            for i, _ in enumerate(self.models):
                self.models[i].X_train = X_train[i::3]
                self.models[i].y_train = y_train[i::3]


        self.trained = [False, False, False]
        self.model_pointer = 0

    def train(self, idx):

        self.trained[idx] = True
        self.models[idx].train(step_size = self.models[idx].learning_rate,
                                b = self.models[idx].b, restart = True)

    def train_next(self):
        cycle_cnt = 0
        while(True):
            cycle_cnt += 1
            idx = self.model_pointer
            self.model_pointer += 1
            print('Training model: {}'.format(idx))
            if self.trained[idx]:
                print('Model already trained')
                if cycle_cnt == 3: return 0
                continue
            else:
                break
        self.trained[idx] = True
        self.models[idx].train(step_size = self.models[idx].learning_rate,
                                b = self.models[idx].b, restart = True)

    def predict(self, feat, processed = False):
        if not np.alltrue(self.trained):
            raise Exception('Not all models trained. Call train_next()')

        predictions = []
        for m in self.models:
            predictions.append(m.predict(feat, processed))
        return np.mean(np.array(predictions), axis = 0)

    def predict_from_hdf5(self, path):
        if not np.alltrue(self.trained):
            raise Exception('Not all models trained. Call train_next()')

        predictions = []
        for m in self.models:
            predictions.append(m.predict_from_hdf5(path))
        return np.mean(np.array(predictions), axis = 0)

    def save(self, net_dir, override = False):
        if net_dir[-1] == '/': net_dir = net_dir[:-1]
        for i, model in enumerate(self.models):
            model.save_all(net_dir + '/ensemble_{}'.format(i+1), override)

    def std_predict(self, feat, processed = False):

        if not np.alltrue(self.trained):
            raise Exception('Not all models trained. Call train_next()')

        predictions = []
        for m in self.models:
            predictions.append(m.predict(feat, processed))
        std = np.std(np.array(predictions), axis = 0)

        return std

def load_ensemble_network(net_dir, n, species):

    if net_dir[-1] == '/': net_dir = net_dir[:-1]
    if n > 0:
        model = load_force_model(net_dir + '/ensemble_1/', species)
    ensemble = Ensemble_Network(model, n)
    for i in range(n):
        ensemble.models[i] = load_force_model(net_dir + '/ensemble_{}/'.format(i+1), species)
        ensemble.trained[i] = [True]
    return ensemble
