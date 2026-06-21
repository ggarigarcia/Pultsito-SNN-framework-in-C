#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "arceus.h"
#include "datasets/gifti.h"



/* main.c */
int main(int argc, char *argv[]) {

    // randomize execution
    srand(time(NULL));

    /*
    Load and initialize
    */
    printf(" ============================= \n Loading and initializing data \n ============================= \n");

    // load configuration parameters from input file
    printf(" > Loading configuration file...\n");
    simulation_configuration_t *conf = load_configuration_params_from_toml(argv[1]);
    printf(" > Configuration file loaded!\n\n");
    fflush(stdout);

    // load parcellation data BEFORE network init (required for topology)
    parcellation_t *parcellation_data = NULL;
    if (conf->enable_parcellation) {
        printf(" > Loading parcellation...\n");
        parcellation_data = load_parcellation(conf->parcellation_file);
        if (!parcellation_data) {
            printf(" > Error loading parcellation! Exiting.\n");
            return 1;
        }
        load_parcel_bold(parcellation_data, conf->parcel_bold_file);
        if (!parcellation_data->parcel_bold) {
            printf(" > Error loading BOLD data! Exiting.\n");
            free_parcellation(parcellation_data);
            return 1;
        }
        conf->parcellation = parcellation_data;
        printf(" > Parcellation loaded (%zu parcels, %zu timepoints)\n",
               parcellation_data->n_parcels, parcellation_data->n_timepoints);
        fflush(stdout);
    }

    // initialize network (uses conf->parcellation if parcel_topology is set)
    printf(" > Initializing network...\n");
    GPU_SNN_t *cpu_snn = initialize_network_cpu(conf);
    printf(" > Network initialized!\n");
    fflush(stdout);

    // load dataset (optional – may be null for stimulus-free testing)
    printf(" > Loading dataset... \n");
    GPU_dataset_t *cpu_dataset = NULL;
    if (conf->dataset) {
        cpu_dataset = load_dataset_from_file_cpu(conf->dataset, conf->labels, conf->n_samples, conf);
        if(!cpu_dataset){
            printf(" > Error loading dataset! Exiting.\n");
            fflush(stdout);
            return 1;
        }
        printf(" > Dataset loaded!\n");
    } else {
        printf(" > No dataset provided, running with zero input.\n");
    }
    fflush(stdout);

    // compute number of batches
    size_t n_batches = 1, r_samples = 0;
    size_t b;

    if (cpu_dataset) {
        n_batches = cpu_dataset->n_samples / conf->batch_size;
        r_samples = cpu_dataset->n_samples % conf->batch_size;
        n_batches = r_samples > 0 ? n_batches + 1 : n_batches;
    }

    // [CPU]
#ifndef CUDA

    // copy non-constant snn data for parallel batch simulation
    init_batch_snn(cpu_snn, conf);

    // initialize struct to store batch results
    GPU_results_t **results = initialize_batch_results_array(conf, cpu_snn->n_neurons, conf->batch_size, conf->time_steps, 1, n_batches, cpu_snn->clusters_info);

    // loop over batches and simulate
    for(b = 0; b < n_batches; b++){
        
        if((b+1) % 100 == 0){
            printf(" Simulating batch %zu\n", b+1);
            fflush(stdout);
        }

        simulate_batch_CPU(cpu_snn, cpu_dataset, conf, results[b], b, 0);
    }

#else

    // [GPU]
    cuda_info_t *cuda_info = configure_cuda_simulation(cpu_snn, cpu_dataset, conf);
    printf(" Cuda simulation configured\n");
    fflush(stdout);

    GPU_SNN_t **gpu_snn = cpy_SNN2GPU(cpu_snn, cuda_info, conf);
    GPU_dataset_t **gpu_dataset = cpy_dataset2GPU(cpu_dataset, cuda_info);

    GPU_results_t **results = initialize_batch_results_array(conf, cpu_snn->n_neurons, conf->batch_size, 1, 1, n_batches);

    for(b = 0; b < n_batches; b++){
        
        if((b+1) % 100 == 0){
            printf(" Simulating batch %zu\n", b+1);
            fflush(stdout);
        }

        simulate_batch_GPU(results[b], gpu_snn, gpu_dataset, conf, cuda_info, b);
    }
#endif

    display_cluster_spike_matrices(results, n_batches, conf->time_steps);

    store_number_of_spikes_array(results, conf, cpu_snn->n_neurons, conf->batch_size, n_batches);
    store_generated_spikes_array(results, conf, cpu_snn->n_neurons, conf->batch_size, conf->time_steps, n_batches);

    // parcel comparison
    if (parcellation_data) {
        store_parcel_comparison(results, cpu_snn, parcellation_data, conf, n_batches);
    }

    // cleanup
    if (cpu_dataset)     deallocate_dataset_str(cpu_dataset);
    if (parcellation_data) {
        conf->parcellation = NULL;  // avoid dangling pointer
        free_parcellation(parcellation_data);
    }

    return 0;
}
