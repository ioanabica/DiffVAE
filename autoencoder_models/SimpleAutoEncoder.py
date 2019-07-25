from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model, Sequential
from keras import optimizers
import numpy as np

from autoencoder_models.base.base_AutoEncoder import BaseAutoEncoder


class SimpleAutoEncoder(BaseAutoEncoder):
    def __init__(self, original_dim, latent_dim, hidden_layers_dim, batch_size, epochs, learning_rate):
        BaseAutoEncoder.__init__(self, original_dim=original_dim, latent_dim=latent_dim,
                                 batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
        self.hidden_layers_dim = hidden_layers_dim
        self.build_encoder()
        self.build_decoder()
        self.compile_ae()

    def build_encoder(self):
        # Input placeholder
        self.input_placeholder = Input(shape=(self.original_dim,))
        encoder_layer = self.input_placeholder

        for hidden_dim in self.hidden_layers_dim:
            encoder_layer = Dense(hidden_dim, activation='relu')(encoder_layer)
            encoder_layer = BatchNormalization()(encoder_layer)

        self.bottleneck = BatchNormalization()(Dense(self.latent_dim, activation='relu')(encoder_layer))
        self.encoder = Model(self.input_placeholder, self.bottleneck)

        print (self.encoder.summary())

    def build_decoder_model(self):
        self.decoder_model = Sequential()

        decoder_hidden_layers_dim = list(reversed(self.hidden_layers_dim))

        print decoder_hidden_layers_dim

        if (len(decoder_hidden_layers_dim) == 0):
            self.decoder_model.add(Dense(self.original_dim, activation='sigmoid', input_dim=self.latent_dim))
            self.decoder_model.add(BatchNormalization())

        else:
            print ("Adding multiple layers")
            self.decoder_model.add(Dense(decoder_hidden_layers_dim[0], activation='relu', input_dim=self.latent_dim))
            # self.decoder_model.add(BatchNormalization())

            for index in range(0, len(decoder_hidden_layers_dim) - 1):
                print "Add layer with dimensions"
                print decoder_hidden_layers_dim[index + 1]
                self.decoder_model.add(Dense(decoder_hidden_layers_dim[index + 1],
                                             activation='relu', input_dim=decoder_hidden_layers_dim[index]))
                # self.decoder_model.add(BatchNormalization())

            self.decoder_model.add(Dense(self.original_dim, activation='sigmoid'))

    def build_decoder(self):
        self.build_decoder_model()
        self.x_decoded = self.decoder_model(self.bottleneck)

        decoder_input_placeholder = Input(shape=(self.latent_dim,))
        self.decoder = Model(decoder_input_placeholder, self.decoder_model(decoder_input_placeholder))

        print(self.decoder.summary())

    def compile_ae(self):
        self.ae = Model(self.input_placeholder, self.x_decoded)

        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        self.ae.compile(optimizer=adam_optimizer, loss='binary_crossentropy')

        print (self.ae.summary())

    def train_ae(self, data, encoder_filename, decoder_filename):
        perm = np.random.permutation(data.shape[0])
        train_size = int(0.8 * data.shape[0])

        x_train = data[perm[:train_size]]
        x_val = data[perm[train_size:]]

        self.ae.fit(x_train, x_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_data=(x_val, x_val))

        self.save_encoder(encoder_filename=encoder_filename)
        self.save_decoder(decoder_filename=decoder_filename)
