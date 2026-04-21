#include <stdio.h>
#include <stdlib.h>
#include <nifti/nifti1_io.h>


unsigned char* read_stimuli(const char *nifti_filename){

     
    // Load the NIfTI image

    nifti_image *nim = nifti_image_read(nifti_filename, 1);

    if (nim == NULL) {
        fprintf(stderr, "Error: Unable to read NIfTI file %s\n", nifti_filename);
        exit(1);
    }

    printf("NIfTI file: %s\n", nifti_filename);

    printf("Dimensions: %d x %d x %d x %d (Timepoints: %d)\n", nim->dim[1], nim->dim[2], nim->dim[3], nim->dim[4], nim->dim[5]);

    unsigned char *data = (unsigned char *) nim->data;

    //printf ("%d\n", nim->datatype);


    printf("Numero de dimensiones:%d\n",nim->ndim);
    


   if (nim->datatype ==NIFTI_TYPE_UINT8) {   

     printf("Num voxels:%lu\n",nim->nvox);

     size_t nvox = nim->nvox;  // Total number of voxels (including all timepoints)
        
     for (int t = 0; t < nim->nvox; t++) {

       printf("t = %3d | value = %u\n", t, ((unsigned char *)nim->data)[t]);

     }


     //GEHITU DATU MOTA GEHIAGO
	
    }
   else {
     fprintf(stderr, "Data type not supported.");
   }


   /*int x = 30;
    int y = 40;
    int z = 25;
    

    
     int nx = nim->nx, ny = nim->ny, nz = nim->nz, nt = nim->nt;

     int xyz_index = z * nx * ny + y * nx + x;

     printf("Extracting time series from voxel (%d, %d, %d):\n", x, y, z);
     printf("Timepoints: %d\n", nt);

     for (int t = 0; t < nt; t++) {
       int index = t * nx * ny * nz + xyz_index;

       switch (nim->datatype) {
       case NIFTI_TYPE_FLOAT32:
	 printf("t = %3d | value = %f\n", t, ((float *)nim->data)[index]);
	 break;
       case NIFTI_TYPE_INT16:
	 printf("t = %3d | value = %d\n", t, ((short *)nim->data)[index]);
	 break;
       case NIFTI_TYPE_UINT8:
	 printf("t = %3d | value = %u\n", t, ((unsigned char *)nim->data)[index]);
	 break;
       default:
	 fprintf(stderr, "Unsupported data type: %d\n", nim->datatype);
	 nifti_image_free(nim);
	
       }
       }*/

    
    // Free the NIfTI image structure

    nifti_image_free(nim);

    return data;
}




int main(int argc, char *argv[]) {

  if (argc != 2) {
    printf("Usage: %s <nifti_file>\n", argv[0]);
    return 1;
  }

  unsigned char *input=read_stimuli(argv[1]);
    
}

