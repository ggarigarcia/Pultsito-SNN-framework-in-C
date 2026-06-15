#include <stdio.h>
#include <stdlib.h>

#include "encoders/snn_encoder.h"
#include "toml_c/toml.h"

/*
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
    for(int i = 0; i < 6; i++) {
        fprintf(output_file, "%2f, ", encoding_array[i]);
    }
    fprintf(output_file, "%2f", encoding_array[6]); // last one (koma gabe)

    return 0;
}
*/

int decode_snn(char *array_file, char *snn_conf_file) {

    FILE *af, *snn_cf;
    encoding_t data;
    
    /* Irakurri */
    af = fopen(array_file, "r");
    if(af == NULL) { 
        printf("Error opening array_file: %s\n", array_file);
        return 1;
    }

    float tmp_n_neurons, tmp_n_input_neurons, tmp_n_neurons_medium, tmp_n_clusters;
    fscanf(af, "%f, %f, %f, %f, %f, %f, %f",
           &tmp_n_neurons, &tmp_n_input_neurons, &tmp_n_neurons_medium,
           &data.intra_medium_connectivity, &tmp_n_clusters,
           &data.intra_cluster_connectivity, &data.inter_cluster_connectivity);
    data.n_neurons = (size_t)tmp_n_neurons;
    data.n_input_neurons = (size_t)tmp_n_input_neurons;
    data.n_neurons_medium = (size_t)tmp_n_neurons_medium;
    data.n_clusters = (size_t)tmp_n_clusters;
    fclose(af);


    /* Idatzi (snn_conf_file GENERALA) */
    snn_cf = fopen(snn_conf_file, "w");
    if(snn_cf == NULL) {
        printf("Error opening snn_conf_file: %s\n", snn_conf_file);
        return 1;
    }

    fprintf(snn_cf, "[general]\n");
    fprintf(snn_cf, "\tneuron_type = 1\n");
    fprintf(snn_cf, "\tneurons = %zu\n", data.n_neurons);
    fprintf(snn_cf, "\tinput_neurons = %zu\n", data.n_input_neurons);
    fprintf(snn_cf, "\toutput_neurons = 0\n");
    fprintf(snn_cf, "\tsynapsis = 1245\n");
    fprintf(snn_cf, "\tnetwork_is_separated = 1\n");
    fprintf(snn_cf, "\n");
    fprintf(snn_cf, "[neurons]\n");
    fprintf(snn_cf, "\tv_thres = 1\n");
    fprintf(snn_cf, "\tv_rest = 1\n");
    fprintf(snn_cf, "\tt_refract = 1\n");
    fprintf(snn_cf, "\tresistance = 1\n");
    fprintf(snn_cf, "\n");
    fprintf(snn_cf, "[synapsis]\n");
    fprintf(snn_cf, "\tdelay = 1\n");
    fprintf(snn_cf, "\tweight = 1\n");
    fprintf(snn_cf, "\ttraining_zone = 1\n");
    fprintf(snn_cf, "\n");
    fprintf(snn_cf, "[clusters]\n");
    fprintf(snn_cf, "\thas_clusters = 1\n");
    fprintf(snn_cf, "\tn_clusters = %zu\n", data.n_clusters);
    fprintf(snn_cf, "\tn_neurons_medium = %zu\n", data.n_neurons_medium);
    fprintf(snn_cf, "\tn_neurons_cluster = 60\n");
    fprintf(snn_cf, "\tintra_medium_connectivity = %f\n", data.intra_medium_connectivity);
    fprintf(snn_cf, "\tintra_cluster_connectivity  = %f\n", data.intra_cluster_connectivity);
    fprintf(snn_cf, "\tinter_cluster_connectivity = %f\n", data.inter_cluster_connectivity);

    fclose(snn_cf);

    return 0;
}


// para testing
int main(int argc, char *argv) {

    char *array_file, *snn_conf_file;

    array_file = "/home/ggarc/usb/uni/tfg/Pultsito-SNN-framework-in-C/test/conf/network/network_test.toml";
    snn_conf_file = "";

    encode_snn(array_file, snn_conf_file);
}