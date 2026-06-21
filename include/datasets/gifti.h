#ifndef GIFTIIO_H
#define GIFTIIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/// @brief Stores per-parcel information loaded from preprocessed GIFTI + labels
typedef struct parcellation_t {
    int enabled;                ///< flag: data is loaded and ready
    size_t n_parcels;           ///< number of parcels (columns in BOLD matrix)
    size_t n_vertices;          ///< total number of labelled vertices
    size_t n_timepoints;        ///< number of timepoints (rows in BOLD matrix)
    size_t *n_parcel_vertices;  ///< [n_parcels] vertex count per parcel
    size_t *all_vertices;       ///< [n_vertices] flat array of vertex indices
    float  *parcel_bold;        ///< [n_parcels * n_timepoints] row-major BOLD
} parcellation_t;

/// @brief Load parcellation descriptor from a binary .parc file
/// @param path  path to the binary file produced by preprocess_gifti.py
/// @return allocated parcellation_t (call free_parcellation() to release)
parcellation_t* load_parcellation(const char *path);

/// @brief Load parcel-averaged BOLD data into an already-loaded parcellation
/// @param parc  parcellation (must have n_parcels & n_timepoints set)
/// @param path  path to the binary .bold file
void load_parcel_bold(parcellation_t *parc, const char *path);

/// @brief Free all memory owned by a parcellation_t
void free_parcellation(parcellation_t *parc);

#ifdef __cplusplus
}
#endif

#endif
