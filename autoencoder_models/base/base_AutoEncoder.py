from keras.models import load_model
import numpy as np


def get_weights_latent_genes(decoder_filename):
    decoder = load_model(decoder_filename)
    decoder_weights = []

    sequential_layer = decoder.layers[1]
    indices = np.arange(0, len(sequential_layer.get_weights()), step=2)
    for index in indices:
        decoder_weights.append(sequential_layer.get_weights()[index])

    result = decoder_weights[0]
    for index in range(1, len(decoder_weights)):
        result = np.dot(result, decoder_weights[index])

    return result


class BaseAutoEncoder():
    def __init__(self, original_dim, latent_dim, batch_size, epochs, learning_rate):
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def save_encoder(self, encoder_filename):
        self.encoder.save(encoder_filename)

    def save_decoder(self, decoder_filename):
        self.decoder.save(decoder_filename)

    def restore_encoder(self, encoder_filename):
        self.encoder = load_model(encoder_filename)

    def restore_decoder(self, decoder_filename):
        self.decoder = load_model(decoder_filename)

    def reconstruction(self, input_data, decoder_filename):
        self.restore_decoder(decoder_filename)
        return self.decoder.predict(input_data)

    def dimensionality_reduction(self, input_data, encoder_filename):
        self.restore_encoder(encoder_filename)
        return self.encoder.predict(input_data)
