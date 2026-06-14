#include <stdio.h>
#include <stdlib.h>

#include "encoders/snn_encoder.h"
#include "toml_c/toml.h"


int encode_snn(char *snn_file_path) {

    FILE *snn_file = NULL;
    char errbuf[100];

    toml_table_t *tbl, *tbl_general, *tbl_neurons, *tbl_synapses, *tbl_clusters;

    encoding_t *encoding = NULL;

    float *encoding_array = NULL;


    // obtener fichero de config del snn
    // abrir fichero
    snn_file = fopen(snn_file_path, "r");
    if(snn_file == NULL) printf(" > Error opening the file %s\n", snn_file_path);

    tbl = toml_parse_file(snn_file, errbuf, 100);

    tbl_general = toml_table_table(tbl, "general");
    tbl_neurons = toml_table_table(tbl, "neurons");
    tbl_synapses = toml_table_table(tbl, "synapsis");
    tbl_clusters = toml_table_table(tbl, "clusters");

    encoding = malloc(sizeof(encoding_t));
    if(encoding == NULL) printf(" > Error initializing encoding struct");


    // extraer info del fichero a struct intermedio (encoding_t)
    // general
    encoding->n_neurons = (size_t) toml_table_int(tbl_general, "neurons").u.i;
    encoding->n_input_neurons = (size_t) toml_table_int(tbl_general, "input_neurons").u.i;

    // medium
    encoding->n_neurons_medium = (size_t) toml_table_int(tbl_clusters, "n_neurons_medium").u.i;
    encoding->intra_medium_connectivity = (float) toml_table_double(tbl_clusters, "intra_medium_connectivity").u.d;

    // clusters
    encoding->n_clusters = (size_t) toml_table_int(tbl_clusters, "n_clusters").u.i;
    encoding->intra_cluster_connectivity = (float) toml_table_double(tbl_clusters, "intra_cluster_connectivity").u.d;
    encoding->inter_cluster_connectivity = (float) toml_table_double(tbl_clusters, "inter_cluster_connectivity").u.d;


    // codificar info del struct a array
    // todo mejorar
    encoding_array = malloc(7 * sizeof(float));

    encoding_array[0] = (float) encoding->n_neurons;
    encoding_array[1] = (float) encoding->n_input_neurons;
    encoding_array[2] = (float) encoding->n_neurons_medium;
    encoding_array[3] = encoding->intra_medium_connectivity;
    encoding_array[4] = (float) encoding->n_clusters;
    encoding_array[5] = encoding->intra_cluster_connectivity;
    encoding_array[6] = encoding->inter_cluster_connectivity;


    // guardar array en un fichero
    // todo mejorar
    FILE *output_file = fopen("/home/ggarc/usb/uni/tfg/Pultsito-SNN-framework-in-C/test/out/snn-encoded-array", "w"); 
    for(int i = 0; i < 7; i++) {
        fprintf(output_file, "%2f ", encoding_array[i]);
    }

    return 0;
}

int main(int argc, char *argv) {

    encode_snn("/home/ggarc/usb/uni/tfg/Pultsito-SNN-framework-in-C/test/conf/network/network_test.toml");
}