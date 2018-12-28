""" Module that implements Energy_Network, the machine learned correcting functional (MLCF)
for energies
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from .ml_util import *
try:
    from .preprocessing import *
except ImportError:
    print('Preprocessing module not loaded')
from matplotlib import pyplot as plt
import math
import pickle
from collections import namedtuple
import h5py
import json
from ase.io import read
Dataset = namedtuple("Dataset", "data species")


class Energy_Network():
    """ Machine learned correcting functional (MLCF) for energies

        Parameters
        ----------

            subnets: list of Subnetwork
                each subnetwork belongs to a single atom inside the system
                and computes the atomic contributio to the total energy
    """
    def __init__(self, subnets):


        if not isinstance(subnets, list):
            self.subnets = [subnets]
        else:
            self.subnets = subnets


        self.model_loaded = False
        self.rand_state = np.random.get_state()
        self.graph = None
        self.target_mean = 0
        self.target_std = 1
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None
        self.masks = {}
        self.species_nets = {}
        self.species_nets_names = {}
        self.species_gradients_names = {}
        scale_together(subnets)

    # ========= Network operations ============ #

    def __add__(self, other):
        if isinstance(other, Subnet):
            if not len(self.subnets) == 1:
                raise Exception(" + operator only valid if only one training set contained")
            else:
                self.subnets[0] += [other]
        else:
            raise Exception("Datatypes not compatible")

        return self

    def __mod__(self, other):
        if isinstance(other, Subnet):
            self.subnets += [[other]]
        elif isinstance(other, Energy_Network):
            self.subnets += other.subnets
        else:
            raise Exception("Datatypes not compatible")

        return self

    def reset(self):
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None


    def construct_network(self):
        """ Builds the tensorflow graph from subnets
        """

        cnt = 0
        logits = []
        for subnet in self.subnets:
            if isinstance(subnet,list):
                sublist = []
                for s in subnet:
                    sublist.append(s.get_logits(cnt)[0])
                    cnt += 1
                logits.append(sublist)
            else:
                logits.append(subnet.get_logits(cnt)[0])
                cnt += 1

        return logits

    def get_feed(self, which = 'train', train_valid_split = 0.8, seed = 42):
        """ Return a dictionary that can be used as a feed_dict in tensorflow

        Parameters
        -----------
            which: {'train',test'}
                which part of the dataset is used
            train_valid_split: float
                ratio of train and validation set size
            seed: int
                seed parameter for the random shuffle algorithm, make

        Returns
        --------
            (dictionary, dictionary)
                either (training feed dictionary, validation feed dict.)
                or (testing feed dictionary, None)
        """
        train_feed_dict = {}
        valid_feed_dict = {}
        test_feed_dict = {}

        for subnet in self.subnets:
            if isinstance(subnet,list):
                for s in subnet:
                    train_feed_dict.update(s.get_feed('train', train_valid_split, seed))
                    valid_feed_dict.update(s.get_feed('valid', train_valid_split, seed))
                    test_feed_dict.update(s.get_feed('test', train_valid_split, seed))
            else:
                train_feed_dict.update(subnet.get_feed('train', train_valid_split, seed))
                valid_feed_dict.update(subnet.get_feed('valid', train_valid_split, seed))
                test_feed_dict.update(subnet.get_feed('test', seed = seed))

        if which == 'train':
            return train_feed_dict, valid_feed_dict
        elif which == 'test':
            return test_feed_dict, None


    def get_cost(self):
        """ Build the tensorflow node that defines the cost function

        Returns
        -------
            cost_list: [tensorflow.placeholder]
                list of costs for subnets. subnets
                whose outputs are added together share cost functions
        """
        cost_list = []

        for subnet in self.subnets:
            if isinstance(subnet,list):
                cost = 0
                y_ = self.graph.get_tensor_by_name(subnet[0].y_name)
                log = 0
                for s in subnet:
                    log += self.graph.get_tensor_by_name(s.logits_name)
                cost += tf.reduce_mean(tf.reduce_mean(tf.square(y_-log),0))
            else:
                log = self.graph.get_tensor_by_name(subnet.logits_name)
                y_ = self.graph.get_tensor_by_name(subnet.y_name)
                cost = tf.reduce_mean(tf.reduce_mean(tf.square(y_-log),0))
            cost_list.append(cost)

        return cost_list




    def train(self,
              step_size=0.01,
              max_steps=50001,
              b_=0,
              verbose=True,
              optimizer=None,
              adaptive_rate=False,
              multiplier = 1.0):

        """ Train the master neural network

            Parameters
            ----------
                step_size: float
                    step size for gradient descent
                max_steps: int
                    number of training epochs
                b: float
                    regularization parameter
                verbose: boolean
                    print cost for intermediate training epochs
                optimizer: tf.nn.GradientDescentOptimizer,tf.nn.AdamOptimizer, ...
                adaptive_rate: boolean
                    wether to adjust step_size if cost increases
                                not recommended for AdamOptimizer
                multiplier: list of float
                    multiplier that allow to give datasets more
                    weight than others

            Returns
            --------
            None
        """


        self.model_loaded = True
        if self.graph is None:
            self.graph = tf.Graph()
            build_graph = True
        else:
            build_graph = False

        with self.graph.as_default():

            if self.sess == None:
                sess = tf.Session()
                self.sess = sess
            else:
                sess = self.sess



            # Get number of distinct subnet species
            species = {}
            for net in self.subnets:
                if isinstance(net,list):
                    for net in net:
                        for l,_ in enumerate(net.layers):
                            name = net.species
                            species[name] = 1
                else:
                    for l,_ in enumerate(net.layers):
                        name = net.species
                        species[name] = 1
            n_species = len(species)


            # Build all the required tensors
            b = {}
            if build_graph:
                self.construct_network()
                for s in species:
                    b[s] = tf.placeholder(tf.float32,name = '{}/b'.format(s))
            else:
                for s in species:
                    b[s] = self.graph.get_tensor_by_name('{}/b:0'.format(s))

            cost_list = self.get_cost()
            train_feed_dict, valid_feed_dict = self.get_feed('train')
            cost = 0
            if not isinstance(multiplier, list):
                multiplier = [1.0]*len(cost_list)
            print('multipliers: {}'.format(multiplier))
            for c, m in zip(cost_list, multiplier):
                cost += c*m

            # L2-loss
            loss = 0
            with tf.variable_scope("", reuse=True):
                for net in self.subnets:
                    if isinstance(net,list):
                        for net in net:
                            for l, layer in enumerate(net.layers):
                                name = net.species
                                loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                                        b[name]/layer
                    else:
                        for l, layer in enumerate(net.layers):
                            name = net.species
                            loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                                b[name]/layer

            cost += loss

            for i, s in enumerate(species):
                train_feed_dict['{}/b:0'.format(s)] = b_[i]
                valid_feed_dict['{}/b:0'.format(s)] = 0


            if self.optimizer == None:
                if optimizer == None:
                    self.optimizer = tf.train.AdamOptimizer(learning_rate = step_size)
                else:
                    self.optimizer = optimizer

            train_step = self.optimizer.minimize(cost)

            # Workaround to load the AdamOptimizer variables
            if not self.checkpoint_path == None:
                saver = tf.train.Saver()
                saver.restore(self.sess,self.checkpoint_path)
                self.checkpoint_path = None

            initialize_uninitialized(self.sess)

            self.initialized = True

            train_writer = tf.summary.FileWriter('./log/',
                                      self.graph)
            old_cost = 1e8

            for _ in range(0,max_steps):

                sess.run(train_step,feed_dict=train_feed_dict)

                if _%int(max_steps/100) == 0 and adaptive_rate == True:
                    new_cost = sess.run(tf.sqrt(cost),
                        feed_dict=train_feed_dict)

                    if new_cost > old_cost:
                        step_size /= 2
                        print('Step size decreased to {}'.format(step_size))
                        train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
                    old_cost = new_cost

                if _%int(max_steps/10) == 0 and verbose:
                    print('Step: ' + str(_))
                    print('Training set loss:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=train_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost-loss),feed_dict=train_feed_dict)))
                    print('Validation set loss:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=valid_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost),feed_dict=valid_feed_dict)))
                    print('--------------------')
                    print('L2-loss: {}'.format(sess.run(loss,feed_dict=train_feed_dict)))

    def predict(self, features, species, use_masks = False, return_gradient = False):
        """ Get predicted energies

        Parameters
        ----------
        features: np.ndarray
            input features
        species: str
            predict atomic contribution to energy for this species
        use_masks: bool
             whether masks should be applied to the provided features
        return_gradient: bool
            instead of returning energies, return gradient of network
            w.r.t. input features

        Returns
        -------
        np.ndarray
            predicted energies or gradient

        """
        species = species.lower()
        if features.ndim == 2:
            features = features.reshape(-1,1,features.shape[1])
        else:
            raise Exception('features.ndim != 2')
        if use_masks:
            features = features[:,:,self.masks[species]]

        ds = Dataset(features, species)
        targets = np.zeros(features.shape[0])

        if not species in self.species_nets:
            self.species_nets[species] = Subnet()
            found = False

            snet = self.species_nets[species]
            for s in self.subnets:
                if found == True:
                    break
                if isinstance(s,list):
                    for s2 in s:
                        if s2.species == ds.species:
                            snet.scaler = s2.scaler
                            snet.layers = s2.layers
                            snet.targets = s2.targets
                            snet.activations = s2.activations
                            print("Sharing scaler with species " + s2.species)
                            found = True
                            break
                else:
                    if s.species == ds.species:
                        snet.scaler = s.scaler
                        snet.layers = s.layers
                        snet.targets = s.targets
                        snet.activations = s.activations
                        print("Sharing scaler with species " + s.species)
                        break

        snet = self.species_nets[species]
        snet.add_dataset(ds, targets, test_size=0.0)
        if not self.model_loaded:
            raise Exception('Model not loaded!')
        else:
            with self.graph.as_default():
                if species in self.species_nets_names:
                    logits = self.graph.get_tensor_by_name(self.species_nets_names[species])
                    gradients = self.graph.get_tensor_by_name(self.species_gradients_names[species])
                else:
                    logits, x, _ = snet.get_logits(1)
                    gradients = tf.gradients(logits, x)[0].values
                    self.species_nets_names[species] = logits.name
                    self.species_gradients_names[species] = gradients.name
                sess = self.sess
                energies = sess.run(logits, feed_dict=snet.get_feed(which='train',
                     train_valid_split=1.0))
                if return_gradient:
                    grad = sess.run(gradients, feed_dict=snet.get_feed(which='train',
                     train_valid_split=1.0))[0]
                    grad = grad/np.sqrt(snet.scaler.var_).reshape(1,-1)
                    energies = (energies, grad)

                return energies

    def get_energies(self, summarize = True, which = 'train'):
        """ Uses trained model on training or test sets

        Parameters
        -----------
        which: str
            {'train','test'} which set logits are computed for

        Returns
        --------
         list of numpy.ndarray
            resulting energies grouped by independent subnet datasets
        """

        if not self.model_loaded:
            raise Exception('Model not loaded!')
        else:
            with self.graph.as_default():

                logits_list = self.construct_network()

                sess = self.sess
                feed_dict, _ = self.get_feed(train_valid_split = 1.0, which = which)

                return_list = []

                for logits in logits_list:
                    if isinstance(logits,list):
                        result = 0
                        for logits in logits:
                            if summarize:
                                result += sess.run(logits,feed_dict=feed_dict)
                            else:
                                return_list.append(sess.run(logits,feed_dict=feed_dict))
                        if summarize:
                            return_list.append(result)
                    else:
                        return_list.append(sess.run(logits,feed_dict=feed_dict))
                return return_list

    def save_model(self, path):
        """ Save trained model to path
        """

        if path[-5:] == '.ckpt':
            path = path[:-5]

        with self.graph.as_default():
            sess = self.sess
            saver = tf.train.Saver()
            saver.save(sess,save_path = path + '.ckpt')

    def restore_model(self, path):
        """ Load trained model from path
        """

        if path[-5:] == '.ckpt':
            path = path[:-5]

        self.checkpoint_path = path + '.ckpt'
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            self.construct_network()
            b = tf.placeholder(tf.float32,name = 'b')
            saver = tf.train.Saver()
            saver.restore(sess,path + '.ckpt')
            self.model_loaded = True
            self.sess = sess
            self.graph = g
            self.initialized = True

    def save_all(self, net_dir, override = False):
        """ Saves the model including all subnets and datasets
        using pickle to directory net_dir, if directory exists only
        save if override = True
        """
        try:
            os.mkdir(net_dir)
        except FileExistsError:
            if override:
                print('Overriding...')
                import shutil
                shutil.rmtree(net_dir)
                os.mkdir(net_dir)
            else:
                print('Directory/Network already exists. Network not saved...')
                return None

        # Pickle does not seem to be compatible with tensorflow so just
        # save subnetworks with it
        with open(os.path.join(net_dir,'subnets'),'wb') as file:
            pickle.dump(self.subnets,file)

        to_save = {'mask': self.masks}

        self.save_model(os.path.join(net_dir,'model'))

        pickle.dump(to_save, open(os.path.join(net_dir,'supp'), 'wb'))


    def load_all(self, net_dir):
        """ Loads the model in net_dir including all subnets and datasets
        using pickle
        """

        with open(os.path.join(net_dir,'subnets'),'rb') as file:
            self.subnets = pickle.load(file)

        self.restore_model(os.path.join(net_dir,'model'))
        self.masks = pickle.load(open(os.path.join(net_dir,'supp'), 'rb'))['mask']

class Subnet():
    """ Subnetwork that is associated with one Atom
    """

    seed = 42

    def __init__(self):
        self.species = None
        self.n_copies = 0
        self.rad_param = None
        self.ang_param = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.name = None
        self.constructor = fc_nn_g
        self.logits_name = None
        self.x_name = None
        self.y_name = None
        self.layers = [8] * 3
        self.targets = 1
        self.activations = [tf.nn.sigmoid] * 3

    def __add__(self, other):
        if not isinstance(other,Subnet):
            raise Exception("Incompatible data types")
        else:
            return Energy_Network([[self,other]])

    def __mod__(self, other):
        if not isinstance(other,Subnet):
            raise Exception("Incompatible data types")
        else:
            return Energy_Network([[self],[other]])


    def get_feed(self, which, train_valid_split = 0.8, seed = None):
        """ Return a dictionary that can be used as a feed_dict in tensorflow

        Parameters
        -----------
            which: str,
                {'train', 'valid', 'test'}
                which part of the dataset is used
            train_valid_split: float
                ratio of train and validation set size
            seed: int
                seed parameter for the random shuffle algorithm

        Returns
        --------
            dict
        """
        if seed == None:
            seed = Subnet.seed

        if train_valid_split == 1.0:
            shuffle = False
        else:
            shuffle = True


        if which == 'train' or which == 'valid':

            X_train = np.concatenate([self.X_train[i] for i in range(self.n_copies)],
                axis = 1)

            X_train, X_valid, y_train, y_valid = \
                train_test_split(X_train,self.y_train,
                                 test_size = 1 - train_valid_split,
                                 random_state = seed, shuffle = shuffle)
            X_train, X_valid = reshape_group(X_train, self.n_copies) , \
                               reshape_group(X_valid, self.n_copies)

            if which == 'train':
                return {self.x_name : X_train, self.y_name: y_train}
            else:
                return {self.x_name : X_valid, self.y_name : y_valid}

        elif which == 'test':

            return {self.x_name : self.X_test, self.y_name: self.y_test}



    def get_logits(self, i):
        """ Builds the subnetwork by defining logits and placeholders

        Parameters
        -----------
            i: int
                index to label datasets

        Returns
        ---------
            tensorflow tensors
        """

        with tf.variable_scope(self.name) as scope:
                        try:
                            logits,x,y_ = self.constructor(self, i, np.mean(self.targets), np.std(self.targets))
                        except ValueError:
                            scope.reuse_variables()
                            logits,x,y_ = self.constructor(self, i, np.mean(self.targets), np.std(self.targets))

        self.logits_name = logits.name
        self.x_name = x.name
        self.y_name = y_.name
        return logits, x, y_

    def save(self, path):
        """ Use pickle to save the subnet to path
        """

        with open(path,'wb') as file:
            pickle.dump(self,file)

    def load(self, path):
        """ Load subnet from path
        """

        with open(path, 'rb') as file:
            self = pickle.load(file)

    def add_dataset(self, dataset, targets,
        test_size = 0.2, target_filter = None, scale = True):
        """ Adds dataset to the subnetwork.

            Parameters
            -----------
                dataset: dataset
                    contains datasets that will be associated with subnetwork for training and
                    evaluation
                targets: np.ndarray
                    target values for training and evaluation

            Returns
            --------
                None
        """

        if self.species != None:
            if self.species != dataset.species:
                raise Exception("Dataset species does not equal subnet species")
        else:
            self.species = dataset.species

        if not self.n_copies == 0:
            if self.n_copies != dataset.data.shape[1]:
                raise Exception("New dataset incompatible with contained one.")

        self.n_copies = dataset.data.shape[1]
        self.name = dataset.species


        if not test_size == 0.0:
            X_train, X_test, y_train, y_test = \
                train_test_split(dataset.data, targets,
                    test_size= test_size, random_state = Subnet.seed, shuffle = True)
        else:
            X_train = dataset.data
            y_train = targets
            X_test = np.array(X_train)
            y_test = np.array(y_train)

        if scale:
            if self.scaler == None:
                scaler = StandardScaler()
                # scaler = MinMaxScaler(feature_range=(-2,2))
                scaler.fit(X_train.reshape(-1, X_train.shape[2]))
            else:
                scaler = self.scaler

            X_train = scaler.transform(X_train.reshape(-1,
                X_train.shape[2])).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1,
                X_test.shape[2])).reshape(X_test.shape)

            self.scaler = scaler

        self.X_train = X_train.swapaxes(0,1)
        self.X_test = X_test.swapaxes(0,1)
        self.features = X_train.shape[2]

        if y_train.ndim == 1:
            self.y_train = y_train.reshape(-1,1)
            self.y_test = y_test.reshape(-1,1)
        else:
            self.y_train = y_train
            self.y_test = y_test

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

def load_energy_model(path):
    """ Load energy MLCF

        Parameters
        ---------
            path: str
                directory in which energy MLCF is stored

        Returns
        -------
            Energy_Network
    """
    network = Energy_Network([])
    network.load_all(path)
    return network

def build_energy_mlcf(feature_src, target_src, masks = {}, automask_std = 0,
    filters = [], autofilt_percent = 0, test_size = 0.2):

    """ Return a trainable energy MLCF (neural network)

    Parameters
    ----------

        feature_src: list
            list of paths to the hdf5 containing the features
        target_src: list
            list of paths to the csv files containing the target energies
            entries in target_scr and feature_src correspond to each other
        masks: dict,
            containing list booleans; can be used to select which features to use.
            keys specify the atomic species.
            default: use all features

        automask_std: float,
            if mask not set exclude all features whose stdev across dataset
            is smaller than this value

        filters: list,
            containing list of booleans; can be used to exclude datapoints
            in sets (e.g. outliers)

        autofilt_percent: float,
            exclude this percentile of extreme datapoints from set
            (only if filters not set)
        test_size: float,
            relative size of hold_out (test) set
    Returns
    -------

        Energy_Network

    """

    if not len(feature_src) == len(target_src):
        raise Exception('Please provided only one target location for each feature set')

    sets = []
    if len(filters) != len(feature_src):
        filters = [0]*len(feature_src)

    no_mask = False

    for fsrc, tsrc, filter in zip(feature_src, target_src, filters):

        elfs, _ = elf.utils.hdf5_to_elfs_fast(fsrc)

        targets = np.genfromtxt(tsrc, delimiter = ',')

        if not isinstance(filter, list) and not isinstance(filter, np.ndarray):

            percentile_cutoff = autofilt_percent
            lim1 = np.percentile(targets, percentile_cutoff*100)
            lim2 = np.percentile(targets, (1 - percentile_cutoff)*100)
            min_lim, max_lim = min(lim1,lim2), max(lim1,lim2)
            filter = (targets > min_lim) & (targets < max_lim)

        if len(masks) != len(elfs):
            no_mask = True
            for species in elfs:
                feat = np.array(elfs[species])
                masks[species] = (np.std(feat.reshape(-1,feat.shape[-1]),
                        axis = 0) > automask_std)

        targets = targets[filter]
        subnets = []
        for species in elfs:
            feat = np.array(elfs[species])[:,:,masks[species]]
            feat = feat[filter]
            for j in range(feat.shape[1]):
                subnets.append(Subnet())
                subnets[-1].add_dataset(Dataset(feat[:,j:j+1], species),
                    targets, test_size = 0.2)

        sets.append(subnets)
    network = Network(sets)
    network.masks = masks
    return network

def get_energy_filters(target_src, autofilt_percent = 0):
    """ For a given energy target dataset return filter that cutoff the
    upper and lower percentile specified in autofilt_percent

    Parameters
    ----------
    target_src: str
        path of csv file containing energy targets
    autofilt_percent: float
        percentile to cut off

    Returns
    --------
        list of bool
            filters
    """
    filters = []
    for tsrc in target_src:
        targets = np.genfromtxt(tsrc, delimiter = ',')
        percentile_cutoff = autofilt_percent
        lim1 = np.percentile(targets, percentile_cutoff*100)
        lim2 = np.percentile(targets, (1 - percentile_cutoff)*100)
        min_lim, max_lim = min(lim1,lim2), max(lim1,lim2)
        filter = (targets > min_lim) & (targets < max_lim)
        filters.append(filter)
    return filters
