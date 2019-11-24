"""
Code Author: Ioana Bica (ioana.bica95@gmail.com)
"""

import os
import argparse

from autoencoder_models.VAE_models import DisentangledDiffVAE
from data.data_processing import get_gene_expression_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_expression_filename", default='data/Zebrafish/GE_mvg.csv')
    parser.add_argument("--hidden_dimensions", default=[512, 256], type=list)
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

    gene_expression_normalized = get_gene_expression_data(args.gene_expression_filename)

    DiffVAE = DisentangledDiffVAE(original_dim=gene_expression_normalized.shape[1], latent_dim=args.latent_dimension,
                                  hidden_layers_dim=args.h_dims,
                                  batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate)

    DiffVAE.train_vae(gene_expression_normalized,
                      'Saved-Models/Encoders/diffvae_encoder_' + args.model_name + '.h5',
                      'Saved-Models/Decoders/diffvae_decoder_' + args.model_name + '.h5')
