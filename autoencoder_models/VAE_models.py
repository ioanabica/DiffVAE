from keras.layers import Input, Dense, BatchNormalization, Lambda, Dropout
from keras.models import Model, Sequential
from keras import metrics, optimizers
import numpy as np

from keras import metrics
from keras import backend as K
import tensorflow as tf

from autoencoder_models.base.base_VAE import BaseVAE


class DiffVAE(BaseVAE):
    def __init__(self, original_dim, latent_dim, hidden_layers_dim, batch_size, epochs, learning_rate):
        BaseVAE.__init__(self, original_dim=original_dim, latent_dim=latent_dim,
                         batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
        self.hidden_layers_dim = hidden_layers_dim
        self.build_encoder()
        self.build_decoder()
        self.compile_vae()

    def build_encoder(self):
        # Input placeholder
        self.input_placeholder = Input(shape=(self.original_dim,))
        encoder_layer = self.input_placeholder

        for hidden_dim in self.hidden_layers_dim:
            encoder_layer = Dense(hidden_dim, activation='relu')(encoder_layer)
            encoder_layer = BatchNormalization()(encoder_layer)

        self.z_mean = BatchNormalization()(Dense(self.latent_dim, activation='relu')(encoder_layer))
        self.z_log_var = BatchNormalization()(Dense(self.latent_dim, activation='relu')(encoder_layer))

        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        self.encoder = Model(self.input_placeholder, self.z_mean)

    def build_decoder_model(self):
        self.decoder_model = Sequential()

        decoder_hidden_layers_dim = list(reversed(self.hidden_layers_dim))

        print (decoder_hidden_layers_dim)

        if (len(decoder_hidden_layers_dim) == 0):
            self.decoder_model.add(Dense(self.original_dim, activation='sigmoid', input_dim=self.latent_dim))
            # self.decoder_model.add(BatchNormalization())

        else:
            print ("Adding multiple layers")
            self.decoder_model.add(Dense(decoder_hidden_layers_dim[0], activation='relu', input_dim=self.latent_dim))
            # self.decoder_model.add(BatchNormalization())

            for index in range(0, len(decoder_hidden_layers_dim) - 1):
                print ("Add layer with dimensions")
                print (decoder_hidden_layers_dim[index + 1])
                self.decoder_model.add(Dense(decoder_hidden_layers_dim[index + 1],
                                             activation='relu', input_dim=decoder_hidden_layers_dim[index]))
                # self.decoder_model.add(BatchNormalization())

            self.decoder_model.add(Dense(self.original_dim, activation='sigmoid'))

    def build_decoder(self):
        self.build_decoder_model()
        self.x_decoded_mean = self.decoder_model(self.z)

        decoder_input_placeholder = Input(shape=(self.latent_dim,))
        self.decoder = Model(decoder_input_placeholder, self.decoder_model(decoder_input_placeholder))

    def vae_loss_function(self, x, x_decoded_mean):
        recon_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)

        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(recon_loss + kl_loss)

    def compile_vae(self):
        self.vae = Model(self.input_placeholder, self.x_decoded_mean)
        print(self.vae.summary())

        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        self.vae.compile(optimizer=adam_optimizer, loss=self.vae_loss_function)

        print (self.vae.summary())

    def train_vae(self, data, encoder_filename, decoder_filename):
        # Divide into test and train sets
        perm = np.random.permutation(data.shape[0])
        train_size = int(0.8 * data.shape[0])

        x_train = data[perm[:train_size]]
        x_val = data[perm[train_size:]]

        print (x_train.shape)
        print (x_val.shape)

        model = self.vae.fit(x_train, x_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             validation_data=(x_val, x_val))

        self.save_encoder(encoder_filename=encoder_filename)
        self.save_decoder(decoder_filename=decoder_filename)

        del self.vae
        del self.encoder
        del self.decoder
        del self.decoder_model

        return model.history


class DisentangledDiffVAE(DiffVAE):
    def __init__(self, original_dim, latent_dim, hidden_layers_dim, batch_size, epochs, learning_rate):
        DiffVAE.__init__(self, original_dim=original_dim, latent_dim=latent_dim, hidden_layers_dim=hidden_layers_dim,
                         batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

    # Code for computing the mmd adapted from https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder
    def compute_kernel(self, x, y):
        size_x = K.shape(x)[0]
        size_y = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([size_x, 1, dim])), K.stack([1, size_y, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, size_y, dim])), K.stack([size_x, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, dtype='float32'))

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def vae_loss_function(self, x, x_decoded_mean):
        true_samples = tf.random_normal(tf.stack([self.batch_size, self.latent_dim]))
        mmd_loss = self.compute_mmd(true_samples, self.z)

        recon_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)

        return K.mean(recon_loss + mmd_loss)
