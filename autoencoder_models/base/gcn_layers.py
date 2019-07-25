from keras import activations, initializers, constraints
from keras import regularizers
from keras.layers import Dropout
from keras.engine import Layer
import keras.backend as K

####################################################################
#### Code adapted from: https://github.com/tkipf/keras-gcn #########
####################################################################

class GraphConvolution(Layer):
    """Graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self,
                 output_dim,
                 adj_matrix,
                 activation=None,
                 dropout_probability=0.0,
                 use_bias=True,
                 activity_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.adj_matrix = K.variable(adj_matrix)
        self.activation = activations.get(activation)
        self.dropout_probability = dropout_probability
        self.use_bias = use_bias
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim)
        return output_shape

    def build(self, input_shape):
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        self.built = True
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs, mask=None):
        inputs = Dropout(self.dropout_probability)(inputs)
        output = K.dot(inputs, self.kernel)

        output = K.dot(self.adj_matrix, output)

        if self.bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.output_dim,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InnerProduct(Layer):

    def __init__(self, num_nodes, activation='sigmoid', **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(InnerProduct, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.activation = activations.get(activation)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.num_nodes)
        return output_shape

    def call(self, inputs, mask=None):
        x = K.transpose(inputs)
        x = K.dot(inputs, x)
        outputs = self.activation(x)
        return outputs