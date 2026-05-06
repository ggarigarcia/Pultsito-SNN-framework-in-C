#include <stdio.h>
#include <stdlib.h>
#include <nifti/nifti1_io.h>

#define NIFTI_FILE "/home/ggarc/usb/uni/tfg/si-burmuin/data/stimuli/todas_las_images.nii.gz"

/*
    BRIEF:  
        - el programa lee de un fichero nifti concreto y escribe TODOS (~2GB) los datos (pixeles) en un fichero de salida output.txt

    METADATOS:
        - dimensiones: 768 x 768 x 4081
        - datatype: NIFTI_TYPE_UINT8
        - posibles valores de pixel: 0, 128 (fondo de la imagen), 255
    
*/

int main(int argc, char *argv[]) {

    // 1. Cargar la imagen (el segundo argumento '1' indica que lea también los datos)
    nifti_image *nim = nifti_image_read(NIFTI_FILE, 1);

    if (!nim) {
        fprintf(stderr, "Error al leer el archivo NIfTI: %s\n", NIFTI_FILE);
        return 1;
    }

    // 2. Obtener Metadatos
    printf("--- Metadatos de: %s ---\n", nim->fname);
    printf("Dimensiones (nx, ny, nz): %d x %d x %d\n", nim->nx, nim->ny, nim->nz);
    printf("Número de dimensiones totales: %d\n", nim->ndim);
    printf("Resolución (dx, dy, dz): %.2f x %.2f x %.2f mm\n", nim->dx, nim->dy, nim->dz);
    printf("Tipo de dato (datatype): %d\n", nim->datatype);

    if (nim->datatype != NIFTI_TYPE_UINT8) {
        fprintf(stderr, "Error: se esperaba datatype 2 (UINT8), recibido: %d\n", nim->datatype);
        nifti_image_free(nim);
        return 1;
    }

    if (nim->nx != 768 || nim->ny != 768 || nim->nz != 4081) {
        fprintf(stderr, "Error: se esperaban dimensiones 768 x 768 x 4081, recibidas: %d x %d x %d\n",
                nim->nx, nim->ny, nim->nz);
        nifti_image_free(nim);
        return 1;
    }

    // 3. Volcar todos los pixeles a output.txt en bloques por imagen (corte z)
    FILE *out = fopen("output.txt", "w");
    if (!out) {
        perror("Error al crear output.txt");
        nifti_image_free(nim);
        return 1;
    }

    unsigned char *data_ptr = (unsigned char *)nim->data;

    for (int z = 0; z < nim->nz; z++) {
        for (int y = 0; y < nim->ny; y++) {
            for (int x = 0; x < nim->nx; x++) {
                int index = x + y * nim->nx + z * nim->nx * nim->ny;
                if (x == nim->nx - 1) {
                    fprintf(out, "%u", data_ptr[index]);
                } else {
                    fprintf(out, "%u ", data_ptr[index]);
                }
            }
            fprintf(out, "\n");
        }
        fprintf(out, "\n");
    }

    fclose(out);
    printf("Pixeles escritos en output.txt\n");

    // 4. Liberar memoria
    nifti_image_free(nim);

    return 0;
}