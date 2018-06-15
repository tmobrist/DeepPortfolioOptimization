#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn

#used from nnagnt during training to initialize the network, rows = coin number, columns = window size, layers = layers
class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, layers, device):
        #manages CPU options
        tf_config = tf.ConfigProto()
        #intialize session
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        #batch size input
        self.input_num = tf.placeholder(tf.int32, shape=[])
        #input tensor with shape batch size, features, coin numbers, window size, dim standard: 3, 11, 31
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns])
        #previous weights with batch size and coin numbers, previous w: none, 11
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows = rows
        self._columns = columns
        self.output = self._build_network(layers)

    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self, feature_number, rows, columns, layers, device):
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, device)

    # generate the operation, the forward computation
    def _build_network(self, layers):
        #restructuring input tensor
        #network2 = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        network = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        #network, volume = tf.split(network2, [4, 1], 3)
        # volume = network2[:, :, :, 4:5]
        #volume = network2[:, :, :, 0:6]

        # [batch, assets, window, features]

        decay = 0.999
        epsilon = 1e-3
        #scale = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]))
        #beta = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]))
        #pop_mean = tf.Variable(tf.zeros([1, network.get_shape()[1], 1, network.get_shape()[-1]]), trainable=False)
        #pop_var = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]), trainable=False)
        #is_training = tflearn.get_training_mode()

        #if is_training:
        #    batch_mean, batch_var = tf.nn.moments(network, [0, 2], keep_dims=True)
        #    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        #    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        #    with tf.control_dependencies([train_mean, train_var]):
        #        network = tf.nn.batch_normalization(network, batch_mean, batch_var, beta, scale, epsilon)
        #else:
        #    network = tf.nn.batch_normalization(network, pop_mean, pop_var, beta, scale, epsilon)
        #else:


        network = network / network[:, :, -1, 0, None, None]

        #volume = volume / (volume[:, :, -1, 0, None, None]+epsilon)
        """
        
        """




        #tf.truediv(network[:, :, :, :-1, None], network[:, :, -1, None, :-1, None])



        for layer_number, layer in enumerate(layers):
            if layer["type"] == "Volume_LSTM":
                volume = tf.transpose(volume, [0, 2, 3, 1])

                # [batch, window, features, assets]

                resultlist = []
                reuse = False
                for i in range(self._rows):
                    result = tflearn.layers.lstm(volume[:, :, :, i],
                                                 1,
                                                 dropout=(0.8, 0.8),
                                                 reuse=reuse,
                                                 scope="lstm")
                    reuse = True
                    resultlist.append(result)
                volume = tf.stack(resultlist)
                volume = tf.transpose(volume, [1, 0, 2])
                volume = tf.reshape(volume, [-1, self._rows, 1, 1])
            elif layer["type"] == "Volume_Conv":
                width = volume.get_shape()[2]
                volume = tflearn.layers.conv_2d(volume, 1,
                                                [1, width],
                                                [1, 1],
                                                "valid",
                                                "relu",
                                                regularizer="L2",
                                                weight_decay=5e-09)
            elif layer["type"] == "DenseLayer":
                network = tflearn.layers.core.fully_connected(network,
                                                              int(layer["neuron_number"]),
                                                              layer["activation_function"],
                                                              regularizer=layer["regularizer"],
                                                              weight_decay=layer["weight_decay"] )
            elif layer["type"] == "DropOut":
                network = tflearn.layers.core.dropout(network, layer["keep_probability"])
            #conv2d over window with output: [batch, assets, new window, filter number]
            elif layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 [1, width],
                                                 [1, 1],
                                                 "valid",
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])

            #Normalize activations of the previous layer at each batch based on Sergey Ioffe, Christian Szegedy. 2015
            elif layer["type"] == "Batch_Normalization2":
                scale2 = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]))
                beta2 = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]))
                pop_mean2 = tf.Variable(tf.zeros([1, network.get_shape()[1], 1, network.get_shape()[-1]]), trainable=False)
                pop_var2 = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]), trainable=False)
                is_training = tflearn.get_training_mode()

                if is_training:
                    batch_mean2, batch_var2 = tf.nn.moments(network, [0, 2], keep_dims=True)
                    train_mean2 = tf.assign(pop_mean2, pop_mean2 * decay + batch_mean2 * (1 - decay))
                    train_var2 = tf.assign(pop_var2, pop_var2 * decay + batch_var2 * (1 - decay))
                    with tf.control_dependencies([train_mean2, train_var2]):
                        network = tf.nn.batch_normalization(network, batch_mean2, batch_var2, beta2, scale2, epsilon)
                else:
                    network = tf.nn.batch_normalization(network, pop_mean2, pop_var2, beta2, scale2, epsilon)

            # Normalize activations of the previous layer at each batch based on Sergey Ioffe, Christian Szegedy. 2015
            elif layer["type"] == "Batch_Normalization3":
                scale3 = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]))
                beta3 = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]))
                pop_mean3 = tf.Variable(tf.zeros([1, network.get_shape()[1], 1, network.get_shape()[-1]]), trainable=False)
                pop_var3 = tf.Variable(tf.ones([1, network.get_shape()[1], 1, network.get_shape()[-1]]), trainable=False)
                is_training = tflearn.get_training_mode()

                if is_training:
                    batch_mean3, batch_var3 = tf.nn.moments(network, [0, 2], keep_dims=True)
                    train_mean3 = tf.assign(pop_mean3, pop_mean3 * decay + batch_mean3 * (1 - decay))
                    train_var3 = tf.assign(pop_var3, pop_var3 * decay + batch_var3 * (1 - decay))
                    with tf.control_dependencies([train_mean3, train_var3]):
                        network = tf.nn.batch_normalization(network, batch_mean3, batch_var3, beta3, scale3, epsilon)
                else:
                    network = tf.nn.batch_normalization(network, pop_mean3, pop_var3, beta3, scale3, epsilon)

            #conv2d over features with output: [batch, new features, new window, filter number]
            elif layer["type"] == "ConvLayer_over_features":
                network = tf.transpose(network, [0, 3, 2, 1])
                # [batch, features, window, assets]
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 allint(layer["filter_shape"]),
                                                 allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
            elif layer["type"] == "ConvLayer":
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 allint(layer["filter_shape"]),
                                                 allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
            elif layer["type"] == "MaxPooling":
                network = tflearn.layers.conv.max_pool_2d(network, layer["strides"])
            elif layer["type"] == "AveragePooling":
                network = tflearn.layers.conv.avg_pool_2d(network, layer["strides"])
            elif layer["type"] == "LocalResponseNormalization":
                network = tflearn.layers.normalization.local_response_normalization(network)
            elif layer["type"] == "FullyCon_WithW":
                network = tflearn.flatten(network)
                network = tf.concat([network, self.previous_w], axis=1)
                network = tflearn.fully_connected(network, self._rows+1,
                                                  activation="relu",
                                                  regularizer=layer["regularizer"],
                                                  weight_decay=layer["weight_decay"])
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, 1, [1, width], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                network = network[:, :, 0, 0]
                btc_bias = tf.ones((self.input_num, 1))
                network = tf.concat([btc_bias, network], 1)
                network = tflearn.layers.core.activation(network, activation="softmax")
            elif layer["type"] == "Output_WithW":
                network = tflearn.fully_connected(network,   self._rows+1,
                                                  activation="softmax",
                                                  regularizer=layer["regularizer"],
                                                  weight_decay=layer["weight_decay"])
            elif layer["type"] == "EIIE_Output_WithW":
                width = network.get_shape()[2]
                #window length
                height = network.get_shape()[1]
                #asset length
                features = network.get_shape()[3]
                #feature length
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                #volume = tf.reshape(volume, [-1, int(height), 1, 1])
                #network = tf.concat([network, w, volume], axis=3)
                network = tf.concat([network, w], axis=3)
                #network = tf.concat([volume, w], axis=2)
                network = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                network = network[:, :, 0, 0]
                btc_bias = tf.zeros((self.input_num, 1))
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                network = tflearn.layers.core.activation(network, activation="softmax")

            elif layer["type"] == "EIIE_LSTM" or\
                            layer["type"] == "EIIE_RNN":
                network = tf.transpose(network, [0, 2, 3, 1])
                #[batch, window, features, assets]
                resultlist = []
                reuse = False
                for i in range(self._rows):
                    if i > 0:
                        reuse = True
                    if layer["type"] == "EIIE_LSTM":
                        result = tflearn.layers.lstm(network[:, :, :, i],
                                                     int(layer["neuron_number"]),
                                                     dropout=layer["dropouts"],
                                                     scope="lstm"+str(layer_number),
                                                     reuse=reuse)
                    else:
                        result = tflearn.layers.simple_rnn(network[:, :, :, i],
                                                           int(layer["neuron_number"]),
                                                           dropout=layer["dropouts"],
                                                           scope="rnn"+str(layer_number),
                                                           reuse=reuse)
                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, self._rows, 1, int(layer["neuron_number"])])
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

# Normalization of the input data
def batch_norm(x, is_training, decay=0.999):
    if x.get_shape()[-1] == 2 or x.get_shape()[-1] == 5:
        epsilon = 1e-3
        scale = tf.Variable(tf.ones([1, x.get_shape()[1], 1, x.get_shape()[-1]]))
        beta = tf.Variable(tf.ones([1, x.get_shape()[1], 1, x.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([1, x.get_shape()[1], 1, x.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([1, x.get_shape()[1], 1, x.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 2], keep_dims=True)
            print(batch_mean)
            print(batch_var)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            print(train_mean)
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            print(train_var)
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)
    else:
        x = x / x[:, :, -1, 0, None, None]
        return x

