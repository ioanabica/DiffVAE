import pandas as pd
from sklearn import preprocessing


def scale_gene_expression_df(data_df):
    return preprocessing.MinMaxScaler().fit_transform(data_df)


def get_gene_expression_data(gene_expression_filename):
    gene_data_df = pd.read_csv(gene_expression_filename, header=None)
    gene_expression_normalized = scale_gene_expression_df(gene_data_df)

    return gene_expression_normalized


def get_zebrafish_gene_expression_data():
    zebrafish_gene_data_df = pd.read_csv("data/Zebrafish/GE_mvg.csv", header=None)
    gene_expression_normalized = scale_gene_expression_df(zebrafish_gene_data_df)

    return gene_expression_normalized


def get_zebrafish_genes():
    zebrafish_genes = pd.read_csv("data/Zebrafish/CV_genes.csv", index_col=False)
    zebrafish_genes = [gene_name.split(',')[1] for gene_name in zebrafish_genes['gene'].values]
    return zebrafish_genes
