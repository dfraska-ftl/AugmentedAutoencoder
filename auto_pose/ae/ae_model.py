'''
Created on Aug 2, 2021

@author: Dave
'''
import tensorflow as tf

class AeEncoder:
    def __init__(
            self, num_filters, strides, kernel_size, batch_norm, variational, latent_space_size):        
        self._batch_norm = batch_norm
        self._variational = variational  
            
        self._filters = []
        self._norm_filters = []
        for filters, strides in zip(num_filters, strides):
            self._filters.append(tf.keras.layers.Conv2D(
                filters, kernel_size, strides,
                padding='same',
                kernel_initializer=tf.initializers.GlorotNormal(),
                activation=tf.nn.relu
            ))
            if batch_norm:
                self._norm_filters.append(
                    tf.keras.layers.BatchNormalization()
                )
        self._z = tf.keras.layers.Dense(
            latent_space_size, activation=None,
            kernel_initializer=tf.initializers.GlorotUniform())
        
        if variational:
            self._epsilon = tf.random.normal(tf.shape(latent_space_size), 0., 1.)
            self._q_sigma = 1e-8 * tf.keras.layers.Dense(
                units=latent_space_size, activation=tf.nn.softplus,
                kernel_initializer=tf.zeros_initializer()
                )
    
    def call(self, inputs, training=False):
        x = inputs
        for idx, filter_ in enumerate(self._filters):
            x = filter_(x)
            if self._batch_norm:
                x = self._norm_filters[idx](x, training=training)
        
        if self._variational:
            x = (self._z + self._q_sigma * self._epsilon)(x)
        else:
            x = self._z(x)
        
        return x

class AeDecoder:
    def __init__(self, latent_code, num_filters, 
                kernel_size, strides, loss, bootstrap_ratio, 
                auxiliary_mask, batch_norm):
        pass
    
    def call(self, inputs, training=False):
        pass

class AeModel(tf.keras.Model):
    def __init__(
            self, encoder_num_filters, encoder_strides,
            kernel_size, batch_norm, variational, latent_space_size):
        super().__init__()
        
        self._encoder = AeEncoder(encoder_num_filters, encoder_strides,
            kernel_size, batch_norm, variational, latent_space_size)
    
    def call(self, inputs, training=False):
        x = self._encoder.call(inputs, training)
        return self._decoder.call(x, training)
        
        #TODO: Add decoder items
        
        # x = self.dense1(inputs)
        # return self.dense2(x)