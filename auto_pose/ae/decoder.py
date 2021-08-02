# -*- coding: utf-8 -*-

import numpy as np

import tensorflow.compat.v1 as tf1

from .utils import lazy_property

class Decoder(object):

    def __init__(self, reconstruction_target, latent_code, num_filters, 
                kernel_size, strides, loss, bootstrap_ratio, 
                auxiliary_mask, batch_norm, is_training=False):
        self._reconstruction_target = reconstruction_target
        self._latent_code = latent_code
        self._auxiliary_mask = auxiliary_mask
        if self._auxiliary_mask:
            self._xmask = None
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._loss = loss
        self._bootstrap_ratio = bootstrap_ratio
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self.reconstr_loss

    @property
    def reconstruction_target(self):
        return self._reconstruction_target

    @lazy_property
    def x(self):
        z = self._latent_code

        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        print(h,w,c)
        layer_dimensions = [ [int(h/np.prod(self._strides[i:])), int(w/np.prod(self._strides[i:]))]  for i in range(len(self._strides))]
        print(layer_dimensions)
        x = tf1.layers.dense(
            inputs=self._latent_code,
            units= layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
            activation=tf1.nn.relu,
            kernel_initializer=tf1.initializers.glorot_uniform()
        )
        if self._batch_normalization:
            x = tf1.layers.batch_normalization(x, training=self._is_training)
        x = tf1.reshape( x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0] ] )

        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            x = tf1.image.resize_nearest_neighbor(x, layer_size)

            x = tf1.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf1.initializers.glorot_normal(),
                activation=tf1.nn.relu
            )
            if self._batch_normalization:
                x = tf1.layers.batch_normalization(x, training=self._is_training)
        
        x = tf1.image.resize_nearest_neighbor( x, [h, w] )

        if self._auxiliary_mask:
            self._xmask = tf1.layers.conv2d(
                    inputs=x,
                    filters=1,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf1.initializers.glorot_normal(),
                    activation=tf1.nn.sigmoid
                )

        x = tf1.layers.conv2d(
                inputs=x,
                filters=c,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf1.initializers.glorot_normal(),
                activation=tf1.nn.sigmoid
            )
        return x

    @lazy_property
    def reconstr_loss(self):
        print(self.x.shape)
        print(self._reconstruction_target.shape)
        if self._loss == 'L2':
            if self._bootstrap_ratio > 1:

                x_flat = tf1.layers.flatten(self.x)
                reconstruction_target_flat = tf1.layers.flatten(self._reconstruction_target)
                l2 = tf1.losses.mean_squared_error (
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf1.losses.Reduction.NONE
                )
                l2_val,_ = tf1.nn.top_k(l2,k=l2.shape[1]//self._bootstrap_ratio)
                loss = tf1.reduce_mean(l2_val)
            else:
                loss = tf1.losses.mean_squared_error (
                    self._reconstruction_target,
                    self.x,
                    reduction=tf1.losses.Reduction.MEAN
                )
        elif self._loss == 'L1':
            if self._bootstrap_ratio > 1:

                x_flat = tf1.layers.flatten(self.x)
                reconstruction_target_flat = tf1.layers.flatten(self._reconstruction_target)
                l1 = tf1.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf1.losses.Reduction.NONE
                )
                print(l1.shape)
                l1_val,_ = tf1.nn.top_k(l1,k=l1.shape[1]/self._bootstrap_ratio)
                loss = tf1.reduce_mean(l1_val)
            else:
                x_flat = tf1.layers.flatten(self.x)
                reconstruction_target_flat = tf1.layers.flatten(self._reconstruction_target)
                l1 = tf1.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf1.losses.Reduction.MEAN
                )
        else:
            print('ERROR: UNKNOWN LOSS ', self._loss)
            exit()
        
        tf1.summary.scalar('reconst_loss', loss)
        if self._auxiliary_mask:
            mask_loss = tf1.losses.mean_squared_error (
                tf1.cast(tf.greater(tf.reduce_sum(self._reconstruction_target,axis=3,keepdims=True),0.0001),tf.float32),
                self._xmask,
                reduction=tf.losses.Reduction.MEAN
            )
            loss += mask_loss

            tf.summary.scalar('mask_loss', mask_loss)

        return loss
