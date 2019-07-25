from keras import backend as K
import tensorflow as tf

from autoencoder_models.base.base_AutoEncoder import BaseAutoEncoder


class BaseVAE(BaseAutoEncoder):
    def __init__(self, original_dim, latent_dim, batch_size, epochs, learning_rate):
        BaseAutoEncoder.__init__(self, original_dim, latent_dim, batch_size, epochs, learning_rate)

    def sampling(self, args, epsilon_std=1.0):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=tf.shape(self.latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
