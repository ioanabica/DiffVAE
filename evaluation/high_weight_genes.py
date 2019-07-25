import pandas as pd

from autoencoder_models.base.base_AutoEncoder import get_weights_latent_genes
from data.data_processing import get_zebrafish_genes


def zebrafish_compute_weights_latent_genes(decoder_filename, num_latent_dimensions, model):
    decoder_weights = get_weights_latent_genes(decoder_filename)
    zebrafish_genes = get_zebrafish_genes()
    weights_latent_genes = pd.DataFrame(decoder_weights, index=range(num_latent_dimensions), columns=zebrafish_genes)
    weights_latent_genes.to_csv('results/HighWeightGenes/zebrafish_latent' +
                                str(num_latent_dimensions) + model + '_to_gene_weights.csv')
    return weights_latent_genes


def zebrafish_compute_high_weight_genes_latent_dim(weights_latent_genes, latent_dimension, model):
    latent_dim_weights = weights_latent_genes.iloc[latent_dimension:latent_dimension + 1].T
    latent_dim_weights = latent_dim_weights.sort_values(by=[latent_dimension], ascending=False)
    latent_dim_weights.to_csv('results/HighWeightGenes/zebrafish_high_weight_genes_for_latent_dimension_' +
                              str(latent_dimension) + model + '.csv')
    return latent_dim_weights

# weights_latent_genes = zebrafish_compute_weights_latent_genes('Saved-Models/Decoders/vae_decoder_zebrafish.h5')                                                         # num_latent_dimensions=50)
# latent_dimension_31 = zebrafish_compute_high_weight_genes_latent_dim(weights_latent_genes, 31)
