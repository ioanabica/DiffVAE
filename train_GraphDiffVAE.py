"""
Code Author: Ioana Bica (ioana.bica95@gmail.com)
"""

import os
import argparse
import numpy as np

from autoencoder_models.GraphDiffVAE import GraphDiffVAE
from data.data_processing import get_gene_expression_data
from data.build_graphs import build_correlation_graph


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_expression_filename", default='data/Zebrafish/GE_mvg.csv')
    parser.add_argument("--hidden_dimensions", default=[512], nargs="*", type=int)
    parser.add_argument("--latent_dimension", default=50, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--model_name", default='graph_test')


    return parser.parse_args()

if __name__ == '__main__':

    args = init_arg()
    if not os.path.exists('results/Graphs'):
        os.mkdir('results/Graphs')

    gene_expression_normalized = get_gene_expression_data(args.gene_expression_filename)

    adj_matrix, initial_node_features = build_correlation_graph(gene_expression_normalized, num_neighbors=2)
    np.save('results/Graphs/input_adj_matrix_' + args.model_name + '.npy', adj_matrix)

    GraphVAE_model=GraphDiffVAE(num_nodes=adj_matrix.shape[0], num_features=initial_node_features.shape[1],
                                adj_matrix=adj_matrix, latent_dim=args.latent_dimension,
                                hidden_layers_dim=args.hidden_dimensions,
                                epochs=args.epochs,
                                learning_rate=args.learning_rate)

    predictions, latent_res = GraphVAE_model.train_vae(initial_node_features, adj_matrix)
    np.save('results/Graphs/predicted_adj_matrix_' + args.model_name + '.npy', predictions)
    np.save('results/Graphs/node_features_' + args.model_name + '.npy', latent_res)

