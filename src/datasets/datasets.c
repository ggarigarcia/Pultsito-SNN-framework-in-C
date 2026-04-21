#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <nifti/nifti1_io.h> 

#include "config/config_loader.h"
#include "datasets/datasets.h"
#include "utils.h"
#include "encoders/image_encoders.h"

/* [PUBLIC] */
GPU_dataset_t* allocate_dataset_str(size_t n_samples, size_t n_features, size_t n_classes, size_t n_spikes){

    GPU_dataset_t *dataset = (GPU_dataset_t*)calloc(1, sizeof(GPU_dataset_t));

    // load general dataset information from configuration struct
    dataset->n_classes = n_classes;
    dataset->n_samples = n_samples;
    dataset->n_features = n_features;
    dataset->n_spikes = n_spikes;

    // allocate memory for arrays
    dataset->sample_offset        = n_samples > 0                   ? (size_t*)malloc(n_samples * sizeof(size_t))              : NULL;
    dataset->n_spikes_per_feature = n_samples > 0 && n_features > 0 ? (size_t*)malloc(n_samples * n_features * sizeof(size_t)) : NULL;
    dataset->feature_offset       = n_samples > 0 && n_features > 0 ? (size_t*)malloc(n_samples * n_features * sizeof(size_t)) : NULL;
    dataset->freq                 = n_samples > 0 && n_features > 0 ? (size_t*)malloc(n_samples * n_features * sizeof(size_t)) : NULL;
    dataset->first_spk            = n_samples > 0 && n_features > 0 ? (size_t*)malloc(n_samples * n_features * sizeof(size_t)) : NULL;
    dataset->spikes               = n_spikes > 0                    ? (size_t*)malloc(n_spikes * sizeof(size_t))               : NULL;

    // return allocated structure
    return dataset;
}

void deallocate_dataset_str(GPU_dataset_t *dataset){
    
    // deallocate internal arrays
    if(dataset->n_spikes_per_feature) free(dataset->n_spikes_per_feature);
    if(dataset->sample_offset)        free(dataset->sample_offset);
    if(dataset->feature_offset)       free(dataset->feature_offset);
    if(dataset->spikes)               free(dataset->spikes);
    if(dataset->freq)                 free(dataset->freq);
    if(dataset->first_spk)            free(dataset->first_spk);

    // deallocate struct
    if(dataset)                        free(dataset);
}

GPU_dataset_t* load_dataset_from_file_cpu(const char *file_name, const char *labels_file_name, size_t n_samples, simulation_configuration_t *conf){

    size_t i, j, l;
    FILE *f = NULL;

    open_file(&f, file_name);

    // allocate memory for dataset (we don't know the number of spikes in the dataset)
    GPU_dataset_t *dataset = allocate_dataset_str(conf->n_samples, conf->input_size, conf->n_classes, 0);

    // count number of spikes in the dataset
    size_t **tmp_spikes = (size_t **)malloc(dataset->n_samples * dataset->n_features * sizeof(size_t*));
    size_t offset = 0;
    size_t n_spikes;
    for(i = 0; i<dataset->n_samples; i++){
        
        // set sample offset
        dataset->sample_offset[i] = offset;

        // loop over features
        for(j = 0; j<dataset->n_features; j++){

            // set feature offset
            dataset->feature_offset[i * dataset->n_features + j] = offset;

            // scan number of spikes of the feature
            fscanf(f, "%zu", &(n_spikes));

            // update offset
            offset += n_spikes;
            dataset->n_spikes += n_spikes;

            // store number of spikes
            dataset->n_spikes_per_feature[i * dataset->n_features + j] = n_spikes;

            // store the spike times of the feature in tmp_spikes
            tmp_spikes[i * dataset->n_features + j] = (size_t*)malloc(n_spikes * sizeof(size_t));
            for(l = 0; l < n_spikes; l++){
                fscanf(f, "%zu", &(tmp_spikes[i * dataset->n_features + j][l]));
            }
        }
    }

    // copy spikes to the dataset struct, compute frequencies and store first spike time
    dataset->spikes = (size_t*)malloc(dataset->n_spikes * sizeof(size_t));

    size_t next_spike = 0;
    for(i = 0; i<dataset->n_samples; i++){

        // loop over features
        for(j = 0; j<dataset->n_features; j++){


            n_spikes = dataset->n_spikes_per_feature[i * dataset->n_features + j];

            for(l = 0; l < n_spikes; l++){

                dataset->spikes[next_spike] = tmp_spikes[i * dataset->n_features + j][l];
                next_spike ++;
            }

            // store first spike and frequency
            if(n_spikes > 0){
                dataset->freq[i * dataset->n_features + j] = conf->max_input_spikes / n_spikes; // spikes each freq time steps
                dataset->first_spk[i * dataset->n_features + j] = dataset->spikes[dataset->feature_offset[i * dataset->n_features + j]];
            }
            else{
                dataset->freq[i * dataset->n_features + j] = 0; // spikes each freq time steps
                dataset->first_spk[i * dataset->n_features + j] = 0;
            }

            // deallocate memory
            free(tmp_spikes[i * dataset->n_features + j]);
        }
    }

    // free temporaly allocated memory
    free(tmp_spikes);

    // return dataset
    return dataset;
}
//TODO: comprobar que lo hace como lo han pedido los teachers:
    // array de arrays, con info del mismo pixel durante los timesteps (recuerda el dibujo de la arbela con el pixel de la esquina de TODAS las imagenes)
    // numero de spike trains = 768 (dimension de imagen)
    // codificacion: 0 (0), 1 (128), 2 (255)
GPU_dataset_t* load_dataset_from_nii(const char *nifti_filename, simulation_configuration_t *conf) {
    int width, height, num_images;
    
    unsigned char* image_buffer = read_stimuli(nifti_filename, &width, &height, &num_images);

    size_t num_pixels = (size_t)width * (size_t)height;
    size_t total_images = (size_t)num_images;
    if (conf->n_samples > 0 && conf->n_samples < total_images) {
        printf(" > Aviso: El NIfTI contiene %zu imágenes, limitando la carga a conf->n_samples=%zu.\n", total_images, conf->n_samples);
        total_images = conf->n_samples;
    } else {
        printf(" > NIfTI cargado con %zu imágenes.\n", total_images);
    }

    size_t total_spikes = 0;

    // Pasada 1: Contar los spikes totales para poder hacer un único allocate de memoria
    for (size_t i = 0; i < total_images; ++i) {
        for (size_t j = 0; j < num_pixels; ++j) {
            int pixel_value = image_buffer[i * num_pixels + j];
            size_t num_spikes = (size_t)((pixel_value / 255.0) * 10);
            total_spikes += num_spikes;
        }
    }

    // Reservar toda la estructura del dataset junta
    GPU_dataset_t *dataset = allocate_dataset_str(total_images, num_pixels, conf->n_classes, total_spikes);
    
    if (total_spikes > 0 && dataset->spikes == NULL) {
        fprintf(stderr, "\nERROR FATAL: No hay suficiente RAM. Malloc falló al intentar reservar memoria para %zu spikes.\n", total_spikes);
        exit(1);
    }
    
    if (total_images > 0 && num_pixels > 0 && (dataset->n_spikes_per_feature == NULL || dataset->feature_offset == NULL || dataset->freq == NULL || dataset->first_spk == NULL)) {
        fprintf(stderr, "\nERROR FATAL: No hay suficiente RAM. Malloc falló al intentar reservar memoria para los metadatos.\n");
        exit(1);
    }

    // Pasada 2: Rellenar la estructura con los instantes de los spikes
    size_t spike_idx = 0;
    for (size_t i = 0; i < total_images; ++i) {
        dataset->sample_offset[i] = spike_idx;
        for (size_t j = 0; j < num_pixels; ++j) {
            dataset->feature_offset[i * num_pixels + j] = spike_idx;
            
            int pixel_value = image_buffer[i * num_pixels + j];
            size_t num_spikes = (size_t)((pixel_value / 255.0) * 10);
            dataset->n_spikes_per_feature[i * num_pixels + j] = num_spikes;
            
            for (size_t k = 0; k < num_spikes; ++k) {
                dataset->spikes[spike_idx++] = k * 10; // Ejemplo: spikes cada 10ms
            }

            if (num_spikes > 0) {
                dataset->freq[i * num_pixels + j] = conf->max_input_spikes / num_spikes;
                dataset->first_spk[i * num_pixels + j] = 0; // k=0 -> 0 * 10 = 0
            } else {
                dataset->freq[i * num_pixels + j] = 0;
                dataset->first_spk[i * num_pixels + j] = 0;
            }
        }
    }

    free(image_buffer);

    return dataset;
}

double get_dataset_size(GPU_dataset_t *dataset){
    
    size_t nS = dataset->n_samples;
    size_t nF = dataset->n_features;
    size_t nSpks = dataset->n_spikes;

    return (

        // general and scalars
        (sizeof(GPU_dataset_t) +
        sizeof(int) + 
        sizeof(size_t) * 4 +

        // arrays
        sizeof(size_t) * (nS + nS * nF * 4 + nSpks)) / 8.0
    );
}

void print_dataset(GPU_dataset_t *dataset){

    size_t i, j, l, next = 0;

    printf(" > Printing dataset: \n");
    for(i = 0; i<dataset->n_samples; i++){

        printf(" > > Sample %zu (offset = %zu)\n", i, dataset->sample_offset[i]);

        // loop over features
        for(j = 0; j<dataset->n_features; j++){

            printf(" > >> Feature %zu (offset = %zu): [", j, dataset->feature_offset[i * dataset->n_features + j]);

            // print spikes in feature
            for(l = 0; l<dataset->n_spikes_per_feature[i * dataset->n_features + j]-1; l++){

                printf("%zu, ", dataset->spikes[next]);
                next++;
            }
            printf("%zu]\n", dataset->spikes[next]);
            next++;
        }
    }
}


unsigned char* read_stimuli(const char *nifti_filename, int *width, int *height, int *num_images){
  
    // Carga la imagen NIfTI
    nifti_image *nim = nifti_image_read(nifti_filename, 1);
    if (nim == NULL) {
        fprintf(stderr, "Error: No se puede leer el fichero NIfTI %s\n", nifti_filename);
        exit(1);
    }

    // Obtiene las dimensiones
    *width = nim->nx;
    *height = nim->ny;

    // Calcular num_images desde el volumen total para prevenir fallos si el NIfTI 
    // almacena el tiempo en dim 4 en lugar de dim 3
    long long total_voxels = nim->nvox;
    if (total_voxels > 0 && nim->nx > 0 && nim->ny > 0) {
        *num_images = (int)(total_voxels / (nim->nx * nim->ny));
    } else {
        // Fallback original
        *num_images = nim->nz; 
    }

    // Comprueba el tipo de datos
    if (nim->datatype != NIFTI_TYPE_UINT8) {
        fprintf(stderr, "Error: Tipo de datos no soportado %d\n", nim->datatype);
        nifti_image_free(nim);
        exit(1);
    }

    // Obtiene el tamaño de los datos
    size_t data_size = nim->nvox * nim->nbyper;
    unsigned char *data = (unsigned char*)malloc(data_size);
    if (data == NULL) {
        fprintf(stderr, "Error: No se puede reservar memoria para los datos de la imagen\n");
        nifti_image_free(nim);
        exit(1);
    }

    // Copia los datos
    memcpy(data, nim->data, data_size);

    // Libera la imagen nifti
    nifti_image_free(nim);

    return data;
}


