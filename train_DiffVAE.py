"""
Code Author: Ioana Bica (ioana.bica95@gmail.com)
"""

import os
import argparse

from autoencoder_models.VAE_models import DisentangledDiffVAE
from data.data_processing import get_zebrafish_gene_expression_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dimension", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=int)
    parser.add_argument("--model_name", default='zebrafish50test')

    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if not os.path.exists('Saved-Models/Encoders/'):
        os.mkdir('Saved-Models/Encoders/')
    if not os.path.exists('Saved-Models/Decoders/'):
        os.mkdir('Saved-Models/Decoders/')

    gene_data = get_zebrafish_gene_expression_data()

    DiffVAE = DisentangledDiffVAE(original_dim=gene_data.shape[1], latent_dim=args.latent_dimension,
                                  hidden_layers_dim=[512, 256],
                                  batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate)

    DiffVAE.train_vae(gene_data,
                      'Saved-Models/Encoders/diffvae_encoder_' + args.model_name + '.h5',
                      'Saved-Models/Decoders/diffvae_decoder_' + args.model_name + '.h5')
