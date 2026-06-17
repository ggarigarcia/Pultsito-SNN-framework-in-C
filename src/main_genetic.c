#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "arceus.h"
#include "encoders/snn_encoder.h"


int main(int argc, char *argv[]) {

    if(argc < 2) {
        printf("USAGE: %s <genotypes_file>\n");
        return 1;
    }

    srand(time(NULL));

    printf(" -> STARTING GENETIC ALGORITHM FOR SNN <-\n\n");

    printf(" ============================= \n Loading and initializing data \n ============================= \n");

    // leer fichero de genotipos y obtener info
    char *snn_file = "test/out/snn_conf_general.toml";
    int ret = decode_to_file(argv[1], snn_file);
    if(ret != 0) {
        printf("Error: decode_to_file. Exiting\n");
        return 1;
    }

    // crear network on the fly
    printf(" > Initializing network...\n");
    

    printf(" > Network initialized!\n\n");
    fflush(stdout);

    // cargar dataset

    // simular network (batches)


    return 0;
}