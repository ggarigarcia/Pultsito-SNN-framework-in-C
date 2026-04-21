#include <stdio.h>
#include <stdlib.h>

#include "datasets/datasets.h"
#include "config/config_loader.h"

/*
    COMPILACION: gcc -g -o test/test_nii test/test_nii.c src/datasets/datasets.c src/encoders/image_encoders.c src/config/config_loader.c lib/toml_c/toml.c src/utils.c src/networks/snn_generator.c src/networks/snn.c src/neuron_models/lif_neuron.c src/neuron_models/neuron_models.c -Iinclude -Ilib -Isrc -I/usr/include/nifti -lniftiio -lznz -lm -fopenmp    
    EJECUCION: ./test/test_nii conf_files/simulation/conf.toml ../si-burmuin/data/stimuli/todas_las_images.nii.gz
*/


int main(int argc, char *argv[]) {
    // Necesitamos al menos el fichero de configuración y el fichero .nii
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <fichero_configuracion.toml> <fichero_video.nii>\n", argv[0]);
        return 1;
    }

    char *conf_file = argv[1];
    char *nii_file = argv[2];

    printf("--- Iniciando prueba de load_dataset_from_nii ---\n");

    // 1. Cargar una configuración de prueba
    printf(" > Cargando fichero de configuración: %s\n", conf_file);
    simulation_configuration_t *conf = load_configuration_params_from_toml(conf_file);
    if (conf == NULL) {
        fprintf(stderr, "Error: No se pudo cargar el fichero de configuración.\n");
        return 1;
    }
    printf(" > Configuración cargada correctamente.\n");

    // 2. Llamar a la función principal a probar
    printf(" > Cargando dataset desde NIfTI: %s\n", nii_file);
    GPU_dataset_t *dataset = load_dataset_from_nii(nii_file, conf);

    // 3. Verificar el resultado
    if (dataset == NULL) {
        printf("\nFALLO: El dataset no se pudo cargar.\n");
        free(conf);
        return 1;
    }

    printf("\nÉXITO: El dataset se ha cargado correctamente.\n");
    printf("  - Número de muestras (imágenes): %zu\n", dataset->n_samples);
    printf("  - Número de características (píxeles): %zu\n", dataset->n_features);
    printf("  - Número total de spikes generados: %zu\n", dataset->n_spikes);
    
    // Descomentar para ver el detalle de los spikes (cuidado, puede ser largo)
    // print_dataset(dataset);

    // 4. Liberar memoria
    printf("\n > Liberando memoria...\n");
    deallocate_dataset_str(dataset);
    free(conf);

    printf("--- Prueba finalizada con éxito ---\n");
    return 0;
}
