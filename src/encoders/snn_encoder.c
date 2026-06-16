#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "encoders/snn_encoder.h"
#include "toml_c/toml.h"

/*
int encode_to_snn(char *array_file, char *snn_conf_file) {

    FILE *snn_file = NULL;
    char errbuf[100];

    toml_table_t *tbl, *tbl_general, *tbl_neurons, *tbl_synapses, *tbl_clusters;

    encoding_t *encoding = NULL;

    float *encoding_array = NULL;


    // obtener fichero de config del snn
    // abrir fichero
    snn_file = fopen(array_file, "r");
    if(snn_file == NULL) printf(" > Error opening the file %s\n", array_file);

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


    // guardar array al final del fichero (una linea por individuo)
    FILE *output_file = fopen("/home/ggarc/usb/uni/tfg/Pultsito-SNN-framework-in-C/test/out/snn-encoded-array", "a");
    fprintf(output_file, "\n");
    for(int i = 0; i < 6; i++) {
        fprintf(output_file, "%f, ", encoding_array[i]);
    }
    fprintf(output_file, "%f\n", encoding_array[6]);


    return 0;
}
*/


static void write_config_block(FILE *f, encoding_t *e) {
    fprintf(f, "[general]\n");
    fprintf(f, "\tneuron_type = 1\n");
    fprintf(f, "\tneurons = %zu\n", e->n_neurons);
    fprintf(f, "\tinput_neurons = %zu\n", e->n_input_neurons);
    fprintf(f, "\toutput_neurons = 0\n");
    fprintf(f, "\tsynapsis = 0\n");
    fprintf(f, "\tnetwork_is_separated = 1\n");
    fprintf(f, "\n");
    fprintf(f, "[neurons]\n");
    fprintf(f, "\tv_thres = 1\n");
    fprintf(f, "\tv_rest = 1\n");
    fprintf(f, "\tt_refract = 1\n");
    fprintf(f, "\tresistance = 1\n");
    fprintf(f, "\n");
    fprintf(f, "[synapsis]\n");
    fprintf(f, "\tdelay = 1\n");
    fprintf(f, "\tweight = 1\n");
    fprintf(f, "\ttraining_zone = 1\n");
    fprintf(f, "\n");
    fprintf(f, "[clusters]\n");
    fprintf(f, "\thas_clusters = 1\n");
    fprintf(f, "\tn_clusters = %zu\n", e->n_clusters);
    fprintf(f, "\tn_neurons_medium = %zu\n", e->n_neurons_medium);
    fprintf(f, "\tn_neurons_cluster = 60\n");
    fprintf(f, "\tintra_medium_connectivity = %f\n", e->intra_medium_connectivity);
    fprintf(f, "\tintra_cluster_connectivity  = %f\n", e->intra_cluster_connectivity);
    fprintf(f, "\tinter_cluster_connectivity = %f\n\n", e->inter_cluster_connectivity);
}


int decode_snn(char *array_file, char *snn_conf_file, size_t line_index) {

    FILE *af, *snn_cf;
    encoding_t encoding_info;
    
    /* Irakurri */
    af = fopen(array_file, "r");
    if(af == NULL) { 
        printf("Error opening array_file: %s\n", array_file);
        return 1;
    }

    /* Saltar line_index lineas */
    char skip_buffer[1024];
    for(size_t i = 0; i < line_index; i++) {
        if(fgets(skip_buffer, sizeof(skip_buffer), af) == NULL) {
            fclose(af);
            return 1;
        }
    }

    float tmp_n_neurons, tmp_n_input_neurons, tmp_n_neurons_medium, tmp_n_clusters;
    int matched = fscanf(af, "%f, %f, %f, %f, %f, %f, %f",
                         &tmp_n_neurons, &tmp_n_input_neurons, &tmp_n_neurons_medium,
                         &encoding_info.intra_medium_connectivity, &tmp_n_clusters,
                         &encoding_info.intra_cluster_connectivity, &encoding_info.inter_cluster_connectivity);
    fclose(af);
    if(matched != 7) return 1;

    encoding_info.n_neurons = (size_t)tmp_n_neurons;
    encoding_info.n_input_neurons = (size_t)tmp_n_input_neurons;
    encoding_info.n_neurons_medium = (size_t)tmp_n_neurons_medium;
    encoding_info.n_clusters = (size_t)tmp_n_clusters;

    /* Idatzi (snn_conf_file GENERALA) */
    snn_cf = fopen(snn_conf_file, "w");
    if(snn_cf == NULL) {
        printf("Error opening snn_conf_file: %s\n", snn_conf_file);
        return 1;
    }
    write_config_block(snn_cf, &encoding_info);
    fclose(snn_cf);

    return 0;
}


int decode_to_snn(char *array_file, char *snn_conf_file) {

    FILE *af, *snn_cf;
    encoding_t encoding_info;
    
    af = fopen(array_file, "r");
    if(af == NULL) { 
        printf("Error opening array_file: %s\n", array_file);
        return 1;
    }

    snn_cf = fopen(snn_conf_file, "w");
    if(snn_cf == NULL) {
        printf("Error opening snn_conf_file: %s\n", snn_conf_file);
        fclose(af);
        return 1;
    }

    float tmp_n_neurons, tmp_n_input_neurons, tmp_n_neurons_medium, tmp_n_clusters;
    while(fscanf(af, "%f, %f, %f, %f, %f, %f, %f",
                 &tmp_n_neurons, &tmp_n_input_neurons, &tmp_n_neurons_medium,
                 &encoding_info.intra_medium_connectivity, &tmp_n_clusters,
                 &encoding_info.intra_cluster_connectivity, &encoding_info.inter_cluster_connectivity) == 7) {
        encoding_info.n_neurons = (size_t)tmp_n_neurons;
        encoding_info.n_input_neurons = (size_t)tmp_n_input_neurons;
        encoding_info.n_neurons_medium = (size_t)tmp_n_neurons_medium;
        encoding_info.n_clusters = (size_t)tmp_n_clusters;

        write_config_block(snn_cf, &encoding_info);
    }

    fclose(af);
    fclose(snn_cf);
    return 0;
}


// para testing
int main() {

    decode_to_snn("test/out/genotypes", "test/out/snn_conf_file");
    
    return 0;
}