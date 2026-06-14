#include <stddef.h>


/**
 * struct que guarda todo lo necesario para codificar una snn
 */
typedef struct encoding_t {

    // general
    size_t n_neurons;
    size_t n_input_neurons;
    
    // medium
    size_t n_neurons_medium;
    float intra_medium_connectivity; // connectivity of medium cluster neurons
    //size_t input_medium_ratio; // nº of input_neuron input connections for each medium neuron 
    
    // clusters
    size_t n_clusters;
    float intra_cluster_connectivity; // array of intra connectivities
    float inter_cluster_connectivity; // array of inter connectivities
    //size_t medium_cluster_ratio; // num conn of each medium neuron to clusters neurons

} encoding_t;



/**
 * codificar snn a array
 * codifica todo lo necesario para generar una snn
 */
int encode_snn(char *snn_file);

/**
 * decodificar array a snn
 */
int decode_snn();