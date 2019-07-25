from keras.layers import Input, Lambda, Average, Concatenate
from keras.models import Model, Sequential
from keras import metrics, optimizers
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
import numpy as np

from autoencoder_models.base.base_VAE import BaseVAE
from autoencoder_models.base.gcn_layers import GraphConvolution, InnerProduct


class GraphDiffVAE(BaseVAE):
    def __init__(self, num_nodes, num_features, adj_matrix, latent_dim, hidden_layers_dim, epochs, learning_rate):
        BaseVAE.__init__(self, original_dim=None, latent_dim=latent_dim,
                         batch_size=1, epochs=epochs, learning_rate=learning_rate)
        self.hidden_layers_dim = hidden_layers_dim
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.adj_matrix = adj_matrix

        self.build_encoder()
        self.build_decoder()
        self.compile_vae()

    def build_encoder(self):
        # Input placeholder
        self.input_placeholder = Input(shape=(self.num_features,))
        self.encoder_layer = self.input_placeholder

        encoder_hidden_layer = GraphConvolution(output_dim=self.hidden_layers_dim[0],
                                         adj_matrix=self.adj_matrix,
                                         activation='relu')(self.encoder_layer)


        self.z_mean = GraphConvolution(output_dim=self.latent_dim,
                                       adj_matrix=self.adj_matrix,
                                       activation='linear')(encoder_hidden_layer)


        self.z_log_var = GraphConvolution(output_dim=self.latent_dim,
                                          adj_matrix=self.adj_matrix,
                                          activation='linear')(encoder_hidden_layer)


        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        self.encoder = Model(self.input_placeholder, self.z_mean)

    def build_decoder(self):
    
        features_concat = Concatenate()([self.encoder_layer, self.z])

        decoder_hidden_layer = GraphConvolution(output_dim=self.hidden_layers_dim[0],
                                                adj_matrix=self.adj_matrix,
                                                activation='relu')(features_concat)

        decoder_z = GraphConvolution(output_dim=self.latent_dim,
                                                adj_matrix=self.adj_matrix,
                                                activation='linear')(decoder_hidden_layer)

        z_combined = Average()([self.z, decoder_z]) # lambda = 0.5
        self.x_decoded_mean = InnerProduct(num_nodes=self.num_nodes)(z_combined)


    def graph_vae_loss_function(self, x, x_decoded_mean):
        x = K.reshape(x, shape=(self.num_nodes*self.num_nodes,))
        x_decoded_mean = K.reshape(x_decoded_mean, [-1])

        norm = self.adj_matrix.shape[0] * self.adj_matrix.shape[0] / \
               float((self.adj_matrix.shape[0] * self.adj_matrix.shape[0] -
                      self.adj_matrix.sum()) * 2)

        recon_loss = norm * metrics.categorical_crossentropy(y_true=x, y_pred=x_decoded_mean)

        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return K.mean(recon_loss + kl_loss)


    def compile_vae(self):
        K.set_learning_phase(1)
        self.graph_vae = Model(self.input_placeholder, self.x_decoded_mean)

        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        self.graph_vae.compile(optimizer=adam_optimizer, loss=self.graph_vae_loss_function)

        print (self.graph_vae.summary())


    def train_vae(self, node_features, adj_mat):
        self.graph_vae.fit(node_features, adj_mat,
                epochs=self.epochs,
                batch_size=self.num_nodes,
                shuffle=False)

        return self.graph_vae.predict(node_features, batch_size=self.num_nodes), \
               self.encoder.predict(node_features, batch_size=self.num_nodes)




