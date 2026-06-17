#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "arceus.h"
#include "encoders/snn_encoder.h"

int main(int argc, char *argv[]) {

    if(argc < 3){
        printf("Usage: %s <genotypes> <simulation_config.toml>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    /*
    Step 1: Read genotype from file (first line)
    */
    printf(" ============================= \n Reading genotype \n ============================= \n");
    printf(" > Reading genotype from %s...\n", argv[1]);
    encoding_t *enc = decode_to_snn(argv[1], 0);
    if(!enc){
        printf(" > Error reading genotype from %s! Exiting.\n", argv[1]);
        fflush(stdout);
        return 1;
    }
    printf(" > Genotype read: %zu neurons, %zu inputs, %zu medium, %zu clusters\n",
           enc->n_neurons, enc->n_input_neurons, enc->n_neurons_medium, enc->n_clusters);
    fflush(stdout);

    /*
    Step 2: Build generator configuration from genotype + defaults
    */
    printf(" > Building generator configuration from genotype...\n");
    generator_conf_t *gen_conf = (generator_conf_t*)calloc(1, sizeof(generator_conf_t));

    // from encoding_t
    gen_conf->n_input       = enc->n_input_neurons;
    gen_conf->n_neurons     = enc->n_neurons;
    gen_conf->n_neurons_medium = enc->n_neurons_medium;
    gen_conf->intra_medium_connectivity = enc->intra_medium_connectivity;
    gen_conf->n_clusters    = enc->n_clusters;
    gen_conf->intra_cluster_connectivity = enc->intra_cluster_connectivity;
    gen_conf->inter_cluster_connectivity = enc->inter_cluster_connectivity;

    // defaults (matching conf_network_test.toml values)
    gen_conf->layered = 0;
    gen_conf->n_output_neurons = 0;
    gen_conf->max_pair_neurons_connections = 1;
    gen_conf->neuron_type = 1; // LIF

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

    free(enc);
    printf(" > Generator configuration built!\n\n");
    fflush(stdout);

    /*
    Step 3: Load simulation configuration
    */
    printf(" ============================= \n Loading simulation configuration \n ============================= \n");
    printf(" > Loading simulation configuration file...\n");
    simulation_configuration_t *sim_conf = load_configuration_params_from_toml(argv[2]);
    printf(" > Simulation configuration loaded!\n\n");
    fflush(stdout);

    /*
    Step 4: Generate network topology
    */
    printf(" ============================= \n Generating network topology \n ============================= \n");
    topology_t topology = generate_clustered_topology(gen_conf);
    topology.neurons = initialize_neurons(gen_conf);
    topology.synapses = initialize_synapses(gen_conf);
    printf(" > Network topology generated!\n");
    fflush(stdout);

    /*
    Step 5: Initialize SNN from generated topology
    */
    printf(" > Building SNN from generated topology...\n");
    GPU_SNN_t *cpu_snn = initialize_network_from_topology(&topology, sim_conf);
    printf(" > SNN initialized!\n");
    fflush(stdout);

    /*
    Step 6: Load dataset
    */
    printf(" > Loading dataset...\n");
    GPU_dataset_t *cpu_dataset = load_dataset_from_file_cpu(sim_conf->dataset, sim_conf->labels, sim_conf->n_samples, sim_conf);
    if(!cpu_dataset){
        printf(" > Error loading dataset! Exiting.\n");
        fflush(stdout);
        return 1;
    }
    printf(" > Dataset loaded!\n");
    fflush(stdout);

    /*
    Step 7: Simulation
    */
    size_t n_batches, r_samples;
    size_t b;

    n_batches = cpu_dataset->n_samples / sim_conf->batch_size;
    r_samples = cpu_dataset->n_samples % sim_conf->batch_size;
    n_batches = r_samples > 0 ? n_batches + 1 : n_batches;

    init_batch_snn(cpu_snn, sim_conf);

    GPU_results_t **results = initialize_batch_results_array(sim_conf, cpu_snn->n_neurons, sim_conf->batch_size, sim_conf->time_steps, 1, n_batches, cpu_snn->clusters_info);

    for(b = 0; b < n_batches; b++){

        if((b+1) % 100 == 0){
            printf(" Simulating batch %zu\n", b+1);
            fflush(stdout);
        }

        simulate_batch_CPU(cpu_snn, cpu_dataset, sim_conf, results[b], b, 0);
    }

    /*
    Step 8: Store results
    */
    display_cluster_spike_matrices(results, n_batches, sim_conf->time_steps);
    store_number_of_spikes_array(results, sim_conf, cpu_snn->n_neurons, sim_conf->batch_size, n_batches);
    store_generated_spikes_array(results, sim_conf, cpu_snn->n_neurons, sim_conf->batch_size, sim_conf->time_steps, n_batches);

    /*
    Step 9: Cleanup
    */
    // topology is on the stack; free its internal arrays manually
    for(b = 0; b < topology.n_neurons; b++){
        if(topology.input_neurons_per_neuron && topology.input_neurons_per_neuron[b])
            free(topology.input_neurons_per_neuron[b]);
    }
    if(topology.input_neurons_per_neuron) free(topology.input_neurons_per_neuron);
    if(topology.neurons.v_thresh) free(topology.neurons.v_thresh);
    if(topology.neurons.v_rest)   free(topology.neurons.v_rest);
    if(topology.neurons.rft_per)  free(topology.neurons.rft_per);
    if(topology.neurons.R)        free(topology.neurons.R);
    if(topology.synapses.w)       free(topology.synapses.w);
    if(topology.synapses.delay)   free(topology.synapses.delay);
    if(topology.synapses.lr)      free(topology.synapses.lr);
    // clusters_info was stolen by initialize_network_from_topology, do not free

    for(b = 0; b < n_batches; b++){
        deallocate_results_str(results[b]);
    }
    free(results);

    deallocate_snn_str(cpu_snn);
    deallocate_dataset_str(cpu_dataset);
    free(gen_conf);
    free(sim_conf);

    printf(" > Done!\n");
    return 0;
}
