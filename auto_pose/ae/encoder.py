# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf1
import numpy as np

from .utils import lazy_property

class Encoder(object):

    def __init__(self, input, latent_space_size, num_filters, kernel_size, strides, batch_norm, is_training=False):
        self._input = input
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self.encoder_out
        self.z
        # self.q_sigma
        # self.sampled_z
        # self.reg_loss
        # self.kl_div_loss

    @property
    def x(self):
        return self._input

    @property
    def latent_space_size(self):
        return self._latent_space_size

    @lazy_property
    def encoder_out(self):
        x = self._input

        for filters, stride in zip(self._num_filters, self._strides):
            padding = 'same'
            x = tf1.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf1.initializers.glorot_normal(),
                activation=tf1.nn.relu
            )
            if self._batch_normalization:
                x = tf1.layers.batch_normalization(x, training=self._is_training)

        encoder_out = tf1.layers.flatten(x)
        
        return encoder_out

    @lazy_property
    def z(self):
        x = self.encoder_out

        z = tf1.layers.dense(
            x,
            self._latent_space_size,   
            activation=None,
            kernel_initializer=tf1.initializers.glorot_uniform()
        )

        return z
    
    @lazy_property
    def q_sigma(self):
        x = self.encoder_out

        q_sigma = 1e-8 + tf1.layers.dense(inputs=x,
                        units=self._latent_space_size,
                        activation=tf1.nn.softplus,
                        kernel_initializer=tf1.zeros_initializer())

        return q_sigma

    @lazy_property
    def sampled_z(self):
        epsilon = tf1.random_normal(tf1.shape(self._latent_space_size), 0., 1.)
        # epsilon = tf1.contrib.distributions.Normal(
        #             np.zeros(self._latent_space_size, dtype=np.float32), 
        #             np.ones(self._latent_space_size, dtype=np.float32))
        return self.z + self.q_sigma * epsilon


    @lazy_property
    def kl_div_loss(self):
        p_z = tf1.distributions.Normal(
            np.zeros(self._latent_space_size, dtype=np.float32), 
            np.ones(self._latent_space_size, dtype=np.float32))
        q_z = tf1.distributions.Normal(self.z, self.q_sigma)

        return tf1.reduce_mean(tf1.distributions.kl_divergence(q_z,p_z))


    @lazy_property
    def reg_loss(self):
        reg_loss = tf1.reduce_mean(tf1.abs(tf1.norm(self.z,axis=1) - tf1.constant(1.)))
        return reg_loss