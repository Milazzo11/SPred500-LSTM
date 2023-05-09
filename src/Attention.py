"""
Custom Attention Layer.

:author: Max Milazzo
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    """
    Attention Layer object definition.
    """
    
    def __init__(self, return_sequences=True):
        """
        Attention Layer initialization.
        
        :param return_sequences: specifies output of 3D tensors when True and
            2D tensors when False
        """
        
        self.return_sequences = return_sequences
        super(Attention,self).__init__()

 
    def build(self, input_shape):
        """
        Builds layer state.
        
        :param input_shape: input shape
        """
        
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1],1), initializer="normal"
        )
        # initialize weights
         
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1],1), initializer="zeros"
        )
        # initialize biases

        super(Attention,self).build(input_shape)

 
    def call(self, x):
        """
        Performs computation and yields output.
        
        :param x: input value
        :return: computation result
        """
        
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        # perform attention computation
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)