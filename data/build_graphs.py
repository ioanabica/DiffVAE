import pandas as pd
from sklearn import preprocessing
import numpy as np
import heapq
from scipy.linalg import sqrtm
from scipy.linalg import inv

from data.data_processing import scale_gene_expression_df


def preprocess_adj_matrix(adj_matrix, symmetric=True):
    adj_matrix = adj_matrix + np.diag(np.ones(adj_matrix.shape[0]))
    D = np.diag(np.dot(adj_matrix, np.ones(adj_matrix.shape[0])))
    D_sqrt_inv = np.linalg.inv(sqrtm(D))
    adj_normalised = np.dot(D_sqrt_inv, np.dot(adj_matrix, D_sqrt_inv))
    return adj_normalised


def build_adj_matrix(node_data_df, num_neighbors):
    adj_matrix = np.zeros(shape=(node_data_df.shape[1], node_data_df.shape[1]), dtype=np.int32)
    #gene_corr = np.absolute(gene_data_df.corr())
    node_corr = node_data_df.corr().as_matrix()
    np.fill_diagonal(node_corr, 0)
    print (node_corr)

    for node_id in range(node_corr.shape[0]):
        node_id_corr = list(node_corr[node_id])
        neighbors = heapq.nlargest(num_neighbors, range(len(node_id_corr)), node_id_corr.__getitem__)
        adj_matrix[node_id][neighbors] = 1

    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def build_correlation_graph(gene_expression_normalized, num_neighbors):
    adj_matrix = build_adj_matrix(pd.DataFrame(gene_expression_normalized.T), num_neighbors)
    node_features = gene_expression_normalized

    return adj_matrix, node_features


def remove_edges_from_stem_cells(adj_matrix, labels):
    for i in range(len(labels)):
        for j in range(len(labels)):
            if adj_matrix[i][j] == 1 and ((labels[i] == 3) or labels[j]==3):
                adj_matrix[i][j] = 0

    return adj_matrix


def build_random_max_neighbors(node_data_df, max_num_neighbors):
    adj_matrix = np.zeros(shape=(node_data_df.shape[1], node_data_df.shape[1]), dtype=np.int32)
    node_corr = node_data_df.corr()

    for node_id in range(300):
        node_id_corr = list(node_corr[node_id])
        neighbors = heapq.nlargest(3, range(len(node_id_corr)), node_id_corr.__getitem__)
        neighbors = np.random.choice(a=neighbors, size=max_num_neighbors, replace=False)
        adj_matrix[node_id][neighbors] = 1

    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def build_cells_graph_zebrafish(num_neighbors):
    zebrafish_gene_data_df = pd.read_csv("data/Zebrafish/GE_mvg.csv", header=None)
    labels = np.load("data/Zebrafish/Annotation_File_Clusters.npy")
    print (labels)

    zebrafish_data_normalized = scale_gene_expression_df(zebrafish_gene_data_df)
    adj_matrix = build_adj_matrix(pd.DataFrame(zebrafish_data_normalized.T), num_neighbors)


    adj_matrix = remove_edges_from_stem_cells(adj_matrix, labels)


    node_features = zebrafish_data_normalized

    return adj_matrix, node_features, labels



