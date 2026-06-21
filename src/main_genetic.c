#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "arceus.h"
#include "encoders/snn_encoder.h"

static void free_topology_internals(topology_t *topology){
    for(size_t i = 0; i < topology->n_neurons; i++){
        if(topology->input_neurons_per_neuron && topology->input_neurons_per_neuron[i])
            free(topology->input_neurons_per_neuron[i]);
    }
    if(topology->input_neurons_per_neuron) free(topology->input_neurons_per_neuron);
    if(topology->neurons.v_thresh) free(topology->neurons.v_thresh);
    if(topology->neurons.v_rest)   free(topology->neurons.v_rest);
    if(topology->neurons.rft_per)  free(topology->neurons.rft_per);
    if(topology->neurons.R)        free(topology->neurons.R);
    if(topology->synapses.w)       free(topology->synapses.w);
    if(topology->synapses.delay)   free(topology->synapses.delay);
    if(topology->synapses.lr)      free(topology->synapses.lr);
}

static int process_genotype(encoding_t *enc, simulation_configuration_t *conf,
                            GPU_dataset_t *cpu_dataset, size_t genotype_idx){

    printf("\n ========== Genotype %zu ========== \n", genotype_idx);
    printf(" > %zu neurons, %zu inputs, %zu medium, %zu clusters\n",
           enc->n_neurons, enc->n_input_neurons, enc->n_neurons_medium, enc->n_clusters);
    fflush(stdout);

    // build generator configuration from encoding + defaults
    generator_conf_t *gen_conf = (generator_conf_t*)calloc(1, sizeof(generator_conf_t));

    gen_conf->n_input       = enc->n_input_neurons;
    gen_conf->n_neurons     = enc->n_neurons;
    gen_conf->n_neurons_medium = enc->n_neurons_medium;
    gen_conf->intra_medium_connectivity = enc->intra_medium_connectivity;
    gen_conf->n_clusters    = enc->n_clusters;
    gen_conf->intra_cluster_connectivity = enc->intra_cluster_connectivity;
    gen_conf->inter_cluster_connectivity = enc->inter_cluster_connectivity;

    gen_conf->layered = 0;
    gen_conf->n_output_neurons = 0;
    gen_conf->max_pair_neurons_connections = 1;
    gen_conf->neuron_type = 1;

    gen_conf->v_thresh_min = 1.0f; gen_conf->v_thresh_max = 1.0f;
    gen_conf->v_rest_min   = 0.0f; gen_conf->v_rest_max   = 0.0f;
    gen_conf->R_min = 1;           gen_conf->R_max = 1;
    gen_conf->rft_per_min = 1;     gen_conf->rft_per_max = 1;

    gen_conf->w_min = 0.15f;       gen_conf->w_max = 0.5f;
    gen_conf->delay_min = 1;       gen_conf->delay_max = 1;
    gen_conf->lr_min = 0;          gen_conf->lr_max = 0;

    gen_conf->input_medium_ratio = 5;
    gen_conf->medium_cluster_ratio = 5;
    gen_conf->store_in_file = 0;

    // generate topology
    topology_t topology = generate_clustered_topology(gen_conf);
    topology.neurons = initialize_neurons(gen_conf);
    topology.synapses = initialize_synapses(gen_conf);

    // build SNN
    GPU_SNN_t *cpu_snn = initialize_network_from_topology(&topology, conf);

    // simulate
    size_t n_batches = cpu_dataset->n_samples / conf->batch_size;
    size_t r_samples = cpu_dataset->n_samples % conf->batch_size;
    n_batches = r_samples > 0 ? n_batches + 1 : n_batches;

    init_batch_snn(cpu_snn, conf);

    GPU_results_t **results = initialize_batch_results_array(
        conf, cpu_snn->n_neurons, conf->batch_size,
        conf->time_steps, 1, n_batches, cpu_snn->clusters_info);

    for(size_t b = 0; b < n_batches; b++){
        if((b+1) % 100 == 0){
            printf(" Simulating batch %zu (genotype %zu)\n", b+1, genotype_idx);
            fflush(stdout);
        }
        simulate_batch_CPU(cpu_snn, cpu_dataset, conf, results[b], b, 0);
    }

    // store results
    display_cluster_spike_matrices(results, n_batches, conf->time_steps);
    store_number_of_spikes_array(results, conf, cpu_snn->n_neurons, conf->batch_size, n_batches);
    store_generated_spikes_array(results, conf, cpu_snn->n_neurons, conf->batch_size, conf->time_steps, n_batches);

    // cleanup per-genotype resources
    free_topology_internals(&topology);

    for(size_t b = 0; b < n_batches; b++){
        deallocate_results_str(results[b]);
    }
    free(results);

    deallocate_snn_str(cpu_snn);
    free(gen_conf);

    return 0;
}

int main(int argc, char *argv[]) {

    if(argc < 3){
        printf("Usage: %s <genotypes> <simulation_config.toml>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    // load simulation configuration once
    printf(" > Loading simulation configuration from '%s'...\n", argv[2]);
    simulation_configuration_t *conf = load_configuration_params_from_toml(argv[2]);
    printf(" > Simulation configuration loaded!\n\n");
    fflush(stdout);

    // load dataset once
    printf(" > Loading dataset from '%s'...\n", conf->dataset);
    GPU_dataset_t *cpu_dataset = load_dataset_from_file_cpu(
        conf->dataset, conf->labels, conf->n_samples, conf);
    if(!cpu_dataset){
        printf(" > Error loading dataset! Exiting.\n");
        fflush(stdout);
        free(conf);
        return 1;
    }
    printf(" > Dataset loaded!\n");
    fflush(stdout);

    // iterate over all genotypes in the file
    printf(" ============================= \n Processing genotypes from '%s'\n ============================= \n", argv[1]);
    for(size_t i = 0; ; i++){

        encoding_t *enc = decode_to_snn(argv[1], i);
        if(!enc) break; // no more lines

        int ret = process_genotype(enc, conf, cpu_dataset, i);
        free(enc);

        if(ret != 0){
            printf(" > Error processing genotype %zu, skipping.\n", i);
            fflush(stdout);
        }
    }

    // cleanup shared resources
    deallocate_dataset_str(cpu_dataset);
    free(conf);

    printf("\n > All genotypes processed!\n");
    return 0;
}
