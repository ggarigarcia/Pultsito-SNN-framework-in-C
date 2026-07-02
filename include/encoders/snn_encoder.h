#ifndef SNN_ENCODER_H
#define SNN_ENCODER_H

#include <stddef.h>

// estructura encargada de guardar info relevante del genotipo
typedef struct encoding_t {
    size_t n_neurons;
    size_t n_input_neurons;
    size_t n_neurons_medium;
    float intra_medium_connectivity;
    size_t n_clusters;
    float intra_cluster_connectivity;
    float inter_cluster_connectivity;

    float fitness;
} encoding_t;

int decode_snn(const char *genotypes, const char *snn_conf_file, size_t line_index);

encoding_t *decode_to_snn(const char *genotypes, size_t index);

encoding_t *read_genotypes(const char *genotypes_file, size_t *n_genotypes);

#endif