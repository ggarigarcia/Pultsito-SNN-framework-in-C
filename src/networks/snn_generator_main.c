#include "networks/snn_generator.h"

#include <stddef.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    
    // randomize exection to create different networks each time
    srand(time(NULL));

    // load configuration file
    generator_conf_t *conf = read_configuration_file(argv[1]);

    // initialize topology
    //topology_t topology = generate_topology(conf); // ORIGINAL
    
    size_t n_clusters = 4;
    float intra_cluster_connectivity = 0.75;
    float inter_cluster_connectivity = 0.25;
    
    size_t n_neurons_medium = 50;
    float intra_medium_connectivity = 0.5;

    topology_t topology = generate_clustered_topology(conf, n_clusters, n_neurons_medium, intra_cluster_connectivity, inter_cluster_connectivity, intra_medium_connectivity);
    

    // initialize neurons and synapses
    topology.neurons = initialize_neurons(conf);
    topology.synapses = initialize_synapses(conf);

    // store generated network
    store_network(&topology, conf, 0);
}