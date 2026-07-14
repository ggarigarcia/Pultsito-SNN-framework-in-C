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

// crear fichero de config de network (usado en snn_generator_main.c) a partir de genotipo
static generator_conf_t *generate_network_conf_file(simulation_configuration_t *conf, encoding_t *enc) {

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
    gen_conf->x = conf->x;
    gen_conf->y = conf->y;
    gen_conf->z = conf->z;
    
    return gen_conf;
}

static float pearson(const float *x, const float *y, size_t n) {
    float mx = 0, my = 0;
    for (size_t i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    float num = 0, dx = 0, dy = 0;
    for (size_t i = 0; i < n; i++) {
        float a = x[i] - mx, b = y[i] - my;
        num += a * b;
        dx  += a * a;
        dy  += b * b;
    }
    float den = sqrtf(dx * dy);
    return (den > 1e-10f) ? num / den : 0.0f;
}

static void calculate_fitness(encoding_t *genotype, GPU_results_t **results, size_t n_batches, size_t batch_size, size_t n_neurons, float *bold_values) {
    
    // info necesaria para obtener valores postprocesados
    size_t B = batch_size;
    size_t nc = results[0]->matrix_t->n_clusters;
    size_t T = results[0]->matrix_t->n_timesteps;
    float *spike_ts = malloc(T * sizeof(float));
    float *bold_ts  = malloc(T * sizeof(float));
    float total_r = 0.0f;

    for (size_t b = 0; b < n_batches; b++) {
        for (size_t c = 0; c < nc; c++) {
            for (size_t t = 0; t < T; t++) {
                spike_ts[t] = (float)results[b]->matrix_t->matrix[t * nc * B + c * B + 0];
                bold_ts[t]  = bold_values[t * nc + c];
            }
            total_r += pearson(spike_ts, bold_ts, T);
        }
    }

    free(spike_ts);
    free(bold_ts);
    genotype->fitness = total_r;
}

// main.c, para el bucle de genotipos
static int process_genotype(encoding_t *genotype, simulation_configuration_t *conf, GPU_dataset_t *cpu_dataset, float *bold_values, size_t genotype_idx){

    //printf(" > > Processing genotype %zu ========== \n", genotype_idx);
    //fflush(stdout);

    // CREAR NETWORK (LO QUE SE HACE EN SNN_GENERATOR_MAIN.C)
    generator_conf_t *gconf = generate_network_conf_file(conf, genotype); // gconf = fichero general de NETWORK
    topology_t t = generate_clustered_topology(gconf);
    t.neurons = initialize_neurons(gconf);
    t.synapses = initialize_synapses(gconf);

    topology_t *topology = malloc(sizeof(topology_t));
    *topology = t;    
    
    // init network
    GPU_SNN_t *cpu_snn = initialize_network_from_topology(topology, conf);
    deallocate_topology_str(topology);
    // printf(" > Network initialized!\n");

    // preparacion para procesamiento de batches
    // TODO: usar constantes en vez de calcular (remember todas_las_images.nii.gz)
    size_t n_batches = cpu_dataset->n_samples / conf->batch_size;
    size_t r_samples = cpu_dataset->n_samples % conf->batch_size;
    n_batches = r_samples > 0 ? n_batches + 1 : n_batches;

    init_batch_snn(cpu_snn, conf);

    GPU_results_t **results = initialize_batch_results_array(
        conf, cpu_snn->n_neurons, conf->batch_size,
        conf->time_steps, 1, n_batches, cpu_snn->clusters_info);

    for(size_t b = 0; b < n_batches; b++){
        /*
        if((b+1) % 100 == 0){
            printf(" Simulating batch %zu (genotype %zu)\n", b+1, genotype_idx);
            fflush(stdout);
        }
        */
        simulate_batch_CPU(cpu_snn, cpu_dataset, conf, results[b], b, 0);
    }

    calculate_fitness(genotype, results, n_batches, conf->batch_size, cpu_snn->n_neurons, bold_values);
    printf(">> >> Genotype %zu, fitness = %2f\n", genotype_idx, genotype->fitness);

    free(gconf);
    free(results);
    free(cpu_snn);

    return 0;
}

// comparar fitness de dos genotipos
static int compare_fitness_desc(const void *a, const void *b) {
    const encoding_t *ea = (const encoding_t *)a;
    const encoding_t *eb = (const encoding_t *)b;
    return (ea->fitness < eb->fitness) - (ea->fitness > eb->fitness);
}

// ordena el array de genotipos y elige "n_best" mejores
static void select_best_genotypes(encoding_t *genotypes, size_t n_genotypes, encoding_t *out, size_t n_best) {
    qsort(genotypes, n_genotypes, sizeof(encoding_t), compare_fitness_desc);
    for (size_t i = 0; i < n_best; i++) out[i] = genotypes[i];
}

// genera descendencia eligiendo aleatoriamente de un padre u otro
static encoding_t crossover(encoding_t *p1, encoding_t *p2) {
    encoding_t c;
    c.n_neurons                  = rand() % 2 ? p1->n_neurons                  : p2->n_neurons;
    c.n_input_neurons            = p1->n_input_neurons; // beti berdina
    c.n_neurons_medium           = rand() % 2 ? p1->n_neurons_medium           : p2->n_neurons_medium;
    c.intra_medium_connectivity  = rand() % 2 ? p1->intra_medium_connectivity  : p2->intra_medium_connectivity;
    c.n_clusters                 = p1->n_clusters; // beti berdina
    c.intra_cluster_connectivity = rand() % 2 ? p1->intra_cluster_connectivity : p2->intra_cluster_connectivity;
    c.inter_cluster_connectivity = rand() % 2 ? p1->inter_cluster_connectivity : p2->inter_cluster_connectivity;
    c.fitness = 0.0f;
    return c;
}

// cambia UNO de los valores (field) modificandolo en un rango DELTA
// TODO guztiak aldatu
static void mutate(encoding_t *g) {

    int field = rand() % 5;
    float delta = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;

    switch (field) {
        case 0: g->intra_medium_connectivity  = fmaxf(0.01f, fminf(1.0f, g->intra_medium_connectivity + delta)); break;
        case 1: g->intra_cluster_connectivity = fmaxf(0.01f, fminf(1.0f, g->intra_cluster_connectivity + delta)); break;
        case 2: g->inter_cluster_connectivity = fmaxf(0.01f, fminf(1.0f, g->inter_cluster_connectivity + delta)); break;
        case 3: {
            int v = (int)g->n_neurons + (rand() % 11 - 5); // valor random
            if (v < 5) v = 5; // minimo / suelo
            g->n_neurons = (size_t)v;
            break;
        }
        case 4: {
            int v = (int)g->n_neurons_medium + (rand() % 5 - 2); // valor random
            if (v < 1) v = 1; // minimo / suelo
            g->n_neurons_medium = (size_t)v;
            break;
        }
    }
}

/* MAIN */

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

    // ! PROVISIONAL: leer valores BOLD generados del script python
    printf("Opening gifti results file\n");
    size_t n_voxels = conf->x * conf->y * conf->z;
    size_t n_timepoints = 135;
    float *bold_values = malloc(n_voxels * n_timepoints * sizeof(float));
    FILE *bold_file = fopen("test/datasets/bold_sumas_sub-S042_ses-001_task-prf_run-01_3x2x2.txt", "r");
    if(!bold_file){
        printf(" > Error opening BOLD file! Exiting.\n");
        fflush(stdout);
        free(conf);
        return 1;
    }
    for(size_t t = 0; t < n_timepoints; t++){
        for(size_t v = 0; v < n_voxels; v++){
            fscanf(bold_file, "%f", &bold_values[t * n_voxels + v]);
        }
    }
    fclose(bold_file);
    printf(" > BOLD file loaded (%zu voxels x %zu timepoints)\n", n_voxels, n_timepoints);
    fflush(stdout);


    // load genotypes
    printf("Reading genotypes from file %s", argv[1]);
    size_t n_genotypes;
    encoding_t *genotypes = read_genotypes(argv[1], &n_genotypes);
    if(genotypes == NULL){
        printf(" > Error loading genotypes! Exiting\n");
        fflush(stdout);
        free(conf);
        deallocate_dataset_str(cpu_dataset);
        return 1;
    }
    printf(" > Genotypes read!\n");


    // ** BUCLE PRINCIPAL **
    // prerrequisitos
    printf("Entering genetic algorithm\n");
    size_t n_best = 5; // TODO cambiar: poner en conf or smth
    encoding_t *best_genotypes = malloc(n_best * sizeof(encoding_t));
    encoding_t *new_genotypes  = malloc(n_genotypes * sizeof(encoding_t));

    
    for(size_t gen = 0; gen < 100; gen++) { // TODO cambiar condicion del loop

        printf("\n >> Entering iteration %zu\n", gen);

        // procesar todos los genotipos del array genotypes -> calcular fitness
        for(size_t j = 0; j < n_genotypes; j++){
            process_genotype(&genotypes[j], conf, cpu_dataset, bold_values, j);
        }

        // mejores "n_best" genotipos
        select_best_genotypes(genotypes, n_genotypes, best_genotypes, n_best);

        // nuevos genotipos: best + descendencia de best mutada
        for(size_t j = 0; j < n_best; j++) genotypes[j] = best_genotypes[j];
        for(size_t j = n_best; j < n_genotypes; j++) {
            encoding_t *p1 = &best_genotypes[rand() % n_best];
            encoding_t *p2 = &best_genotypes[rand() % n_best];
            genotypes[j] = crossover(p1, p2);
            mutate(&genotypes[j]);
        }

        /*
        encoding_t *tmp = genotypes;
        genotypes = new_genotypes;
        new_genotypes = tmp;
        */
    }
    free(best_genotypes);
    //free(new_genotypes);

    /*
    // TODO: calcular fitness fuera del bucle
    printf("\n=== Final genotypes ===\n");
    for(size_t j = 0; j < n_genotypes; j++){
        printf("  #%zu: fitness=%.0f | n_neur=%zu n_in=%zu n_med=%zu n_clust=%zu"
               " | intra_med=%.2f intra_clust=%.2f inter_clust=%.2f\n",
               j, genotypes[j].fitness,
               genotypes[j].n_neurons, genotypes[j].n_input_neurons,
               genotypes[j].n_neurons_medium, genotypes[j].n_clusters,
               genotypes[j].intra_medium_connectivity,
               genotypes[j].intra_cluster_connectivity,
               genotypes[j].inter_cluster_connectivity); // en verdad son los antepenultimos, habria que calcular el fitness de los ultimos fuera
    }
               */

    // cleanup shared resources
    deallocate_dataset_str(cpu_dataset);
    free(bold_values);
    free(conf);

    printf("\n > All genotypes processed!\n");
    return 0;
}
