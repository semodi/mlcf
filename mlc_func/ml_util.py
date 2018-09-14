""" This module contains functions that implement fully connected neural networks
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf

def fc_nn(network):
    """Builds a fully connected neural network

    Input parameters:
    -----------------
        network: subnet; subnet object

    Returns:
    --------
        logits: tensorflow tensor; output layer of neural network
        x: tensorflow placeholder; input layer
        y:  tensorflow placeholder; placeholder for target values
        """


    features = network.features
    layers = network.layers
    targets = network.targets
    activations = network.activations
    namescope = network.name

    n = len(layers)

    W = []
    b = []
    hidden = []

    x = tf.placeholder(tf.float32,[None,features])
    y_ = tf.placeholder(tf.float32,[None,targets])



    W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [features,layers[0]],name='W1'))
    b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[0]],name='b1'))
    hidden.append(activations[0](tf.matmul(x,W[0])+b[0]))

    for l in range(1,n):
        W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [layers[l-1],layers[l]],name='W' + str(l+1)))
        b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[l]],name='b' + str(l+1)))
        hidden.append(activations[l](tf.matmul(hidden[l-1],W[l])+b[l]))

    W.append(tf.get_variable(initializer = tf.truncated_normal_initializer, shape= [layers[n-1],targets],name='W' + str(n+1)))
    b.append(tf.get_variable(initializer = tf.constant_initializer(0), shape = [targets],name='b' + str(n+1)))


    logits = tf.matmul(hidden[n-1],W[n])+b[n]

    return logits,x,y_


def fc_nn_g(network, i, mean = 0, std = 1):
    """Builds a fully connected neural network that consists of network.n_copies
    copies of a subnetwork

    Input parameters:
    -----------------
        network: subnet; subnet object
        i: int; index to label placeholders (for multiple datasets)
        mean: float; mean target value
        std: float; standard deviation of target values

    Returns:
    --------
        logits: tensorflow tensor; output layer of neural network
        x: tensorflow placeholder; input layer
        y:  tensorflow placeholder; placeholder for target values
        """

    features = network.features
    layers = network.layers
    targets = network.targets
    activations = network.activations
    namescope = network.name
    n_copies = network.n_copies
    mean = mean/network.n_copies
    std = std/network.n_copies


    n = len(layers)

    W = []
    b = []
    hidden = []
    x = tf.placeholder(tf.float32,[n_copies,None,features],'x' + str(i))
    y_ = tf.placeholder(tf.float32,[None, 1], 'y_' + str(i))



    W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [features,layers[0]],name='W1'))
    b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[0]],name='b1'))


    for l in range(1,n):
        W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [layers[l-1],layers[l]],name='W' + str(l+1)))
        b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[l]],name='b' + str(l+1)))


    W.append(tf.get_variable(initializer = tf.random_normal_initializer(0,std), shape= [layers[n-1],targets],name='W' + str(n+1)))
    b.append(tf.get_variable(initializer = tf.constant_initializer(mean), shape = [targets],name='b' + str(n+1)))


    for n_g in range(n_copies):
        hidden.append(activations[0](tf.matmul(tf.gather(x,n_g),W[0])/features*10 + b[0]))
        for l in range(0,n-1):
            hidden.append(activations[l+1](tf.matmul(hidden[n_g*n+l],W[l+1])/layers[l]*10 + b[l+1]))

        if n_g == 0:
            logits = tf.matmul(hidden[n_g*n+n-1],W[n])+b[n]
        else:
            logits +=  tf.matmul(hidden[n_g*n+n-1],W[n])+b[n]

    return logits,x,y_

def initialize_uninitialized(sess):
    """ Search graph for uninitialized variables and initialize them
    """
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def get_batch_feed(feed_dict, start, batch_size):

    batch_feed_dict = {}
    for key in feed_dict:
        if feed_dict[key].ndim == 2:
            batch_feed_dict[key] = feed_dict[key][start:batch_size]
        else:
            batch_feed_dict[key] = feed_dict[key][:, start:batch_size, :]

    return batch_feed_dict
