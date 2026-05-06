#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config/config_loader.h"
#include "datasets/datasets.h"
#include <nifti/nifti1_io.h>

static int write_pixels_and_spikes_to_file(const char *output_file,
                                           const simulation_configuration_t *conf,
                                           const GPU_dataset_t *dataset);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <fichero conf simulacion>\n", argv[0]);
        return 1;
    }

    simulation_configuration_t *conf = load_configuration_params_from_toml(argv[1]);
    if (!conf) {
        printf("Error: no se pudo cargar la configuración.\n");
        return 2;
    }
    printf(" > Configuration file loaded!\n\n");

    GPU_dataset_t *dataset = load_dataset_from_nifti_cpu(conf);

    if (!dataset) {
        printf("Error: No se pudo cargar el dataset desde el archivo NIfTI.\n");
        return 2;
    }

    printf("Dataset cargado correctamente.\n");
    printf("n_samples: %zu\n", dataset->n_samples);
    printf("n_features: %zu\n", dataset->n_features);
    printf("n_classes: %zu\n", dataset->n_classes);

    // Muestra los primeros 10 pixeles no grises del primer sample y sus spikes esperados
    nifti_image *nifti = nifti_image_read(conf->dataset, 1);
    if (!nifti) {
        printf("Error: no se pudo volver a abrir el NIfTI para mostrar los pixeles.\n");
        deallocate_dataset_str(dataset);
        return 3;
    }

    if (dataset->n_samples > 0 && dataset->n_features > 0) {
        size_t shown = 0;
        size_t sample_offset = 0;
        uint8_t *pixels = (uint8_t *)nifti->data;

        printf("Primeros 10 pixeles no grises del primer sample:\n");
        for (size_t pixel_index = 0; pixel_index < dataset->n_features && shown < 10; ++pixel_index) {
            uint8_t pixel_value = pixels[sample_offset + pixel_index];

            if (pixel_value == 0 || pixel_value == 255) {
                size_t expected_spikes = (pixel_value == 0) ? 1 : 2;
                size_t generated_spikes = dataset->n_spikes_per_feature[pixel_index];

                printf("pixel[%zu] = %u -> expected spikes: %zu, generated spikes: %zu\n",
                       pixel_index,
                       (unsigned int)pixel_value,
                       expected_spikes,
                       generated_spikes);
                ++shown;
            }
        }

        if (shown == 0) {
            printf("No se encontraron pixeles no grises en el primer sample.\n");
        }
    }

    nifti_image_free(nifti);

    if (!write_pixels_and_spikes_to_file("test/out/nifti_pixels_and_spikes.txt", conf, dataset)) {
        printf("Error: no se pudo escribir el fichero de pixeles y spikes.\n");
        deallocate_dataset_str(dataset);
        return 4;
    }

    // Liberar memoria
    deallocate_dataset_str(dataset);
    return 0;
}

static int write_pixels_and_spikes_to_file(const char *output_file,
                                           const simulation_configuration_t *conf,
                                           const GPU_dataset_t *dataset) {
    FILE *f = fopen(output_file, "w");
    if (!f) {
        perror("Error opening output file");
        return 0;
    }

    nifti_image *nifti = nifti_image_read(conf->dataset, 1);
    if (!nifti) {
        fprintf(stderr, "Error: no se pudo volver a abrir el NIfTI para escribir el fichero.\n");
        fclose(f);
        return 0;
    }

    uint8_t *pixels = (uint8_t *)nifti->data;

    for (size_t sample = 0; sample < dataset->n_samples; ++sample) {
        size_t sample_offset = sample * dataset->n_features;

        for (size_t feature = 0; feature < dataset->n_features; ++feature) {
            fprintf(f, "%4u", (unsigned int)pixels[sample_offset + feature]);
        }
        fputc('\n', f);

        for (size_t feature = 0; feature < dataset->n_features; ++feature) {
            fprintf(f, "%4zu", dataset->n_spikes_per_feature[sample_offset + feature]);
        }
        fputc('\n', f);
        fputc('\n', f);
    }

    nifti_image_free(nifti);
    fclose(f);
    return 1;
}
