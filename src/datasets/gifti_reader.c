#include <stdio.h>
#include <stdlib.h>

#include "datasets/gifti.h"

parcellation_t* load_parcellation(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open parcellation file '%s'\n", path);
        return NULL;
    }

    parcellation_t *parc = calloc(1, sizeof(parcellation_t));

    size_t n_parcels, n_vertices, n_timepoints;

    if (fread(&n_parcels,   sizeof(size_t), 1, f) != 1) goto err;
    if (fread(&n_vertices,  sizeof(size_t), 1, f) != 1) goto err;
    if (fread(&n_timepoints, sizeof(size_t), 1, f) != 1) goto err;

    parc->n_parcels    = n_parcels;
    parc->n_vertices   = n_vertices;
    parc->n_timepoints = n_timepoints;

    parc->n_parcel_vertices = malloc(n_parcels * sizeof(size_t));
    if (fread(parc->n_parcel_vertices, sizeof(size_t), n_parcels, f) != n_parcels)
        goto err;

    parc->all_vertices = malloc(n_vertices * sizeof(size_t));
    if (fread(parc->all_vertices, sizeof(size_t), n_vertices, f) != n_vertices)
        goto err;

    parc->parcel_bold = NULL;  // loaded separately

    fclose(f);
    parc->enabled = 1;
    return parc;

err:
    fprintf(stderr, "Error: malformed parcellation file '%s'\n", path);
    fclose(f);
    free_parcellation(parc);
    return NULL;
}

void load_parcel_bold(parcellation_t *parc, const char *path) {
    if (!parc) return;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open BOLD file '%s'\n", path);
        return;
    }

    size_t n = parc->n_parcels * parc->n_timepoints;

    // free previous data if any
    if (parc->parcel_bold) free(parc->parcel_bold);

    parc->parcel_bold = malloc(n * sizeof(float));
    if (fread(parc->parcel_bold, sizeof(float), n, f) != n) {
        fprintf(stderr, "Error: failed to read BOLD data from '%s'\n", path);
        free(parc->parcel_bold);
        parc->parcel_bold = NULL;
    }

    fclose(f);
}

void free_parcellation(parcellation_t *parc) {
    if (!parc) return;
    if (parc->n_parcel_vertices) free(parc->n_parcel_vertices);
    if (parc->all_vertices)      free(parc->all_vertices);
    if (parc->parcel_bold)       free(parc->parcel_bold);
    free(parc);
}
