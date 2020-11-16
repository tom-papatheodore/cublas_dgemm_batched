#include <stdio.h>
#include <essl.h>
#include <cublas_v2.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                               \
do{                                                                                        \
    cudaError_t cuErr = call;                                                              \
    if(cudaSuccess != cuErr){                                                              \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr)); \
      exit(0);                                                                             \
    }                                                                                      \
}while(0)

int main(int argc, char *argv[])
{
    // Dimension of individual matrices
    int N = 128;

    // Size (in bytes) of individual matrices
    int buffer_size = N * N * sizeof(double);

    // Total number of matrices
    int batch_count = 1024;

    // Set which device to use
    int dev_id = 0;
    cudaErrorCheck( cudaSetDevice(dev_id) );

    // These are host pointer arrays, whose elements will point to individual host matrix buffers
    double **A = (double**)malloc(batch_count * sizeof(double*));
    double **B = (double**)malloc(batch_count * sizeof(double*));
    double **C = (double**)malloc(batch_count * sizeof(double*));

    // These are the individual host matrix buffers
    for(int i=0; i<batch_count; i++){

        A[i] = (double*)malloc(buffer_size);
        B[i] = (double*)malloc(buffer_size);
        C[i] = (double*)malloc(buffer_size);

    }

    // These are host pointer arrays, whose elements will point to individual device matrix buffers
    double **h_d_A = (double**)malloc(batch_count * sizeof(double*));
    double **h_d_B = (double**)malloc(batch_count * sizeof(double*));
    double **h_d_C = (double**)malloc(batch_count * sizeof(double*));

    // These are the individual device matrix buffers
    for(int i=0; i<batch_count; i++){

        cudaErrorCheck( cudaMalloc(&h_d_A[i], buffer_size) );
        cudaErrorCheck( cudaMalloc(&h_d_B[i], buffer_size) );
        cudaErrorCheck( cudaMalloc(&h_d_C[i], buffer_size) );

    }

    // These are device pointer arrays, whose elements will point to individual device matrix buffers
    double **d_A, **d_B, **d_C;
    cudaErrorCheck( cudaMalloc(&d_A, batch_count * sizeof(double*)) );
    cudaErrorCheck( cudaMalloc(&d_B, batch_count * sizeof(double*)) );
    cudaErrorCheck( cudaMalloc(&d_C, batch_count * sizeof(double*)) );

    // Pass values of host pointer arrays to device
    cudaErrorCheck( cudaMemcpy(d_A, h_d_A, batch_count * sizeof(double*), cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_B, h_d_B, batch_count * sizeof(double*), cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_C, h_d_C, batch_count * sizeof(double*), cudaMemcpyHostToDevice) );

    // Max size for random double
    double max_value = 10.0;

    // Allocate memory for test matrices that will serve as the correct answer from the host.
    double *test_A = (double*)malloc(buffer_size);
    double *test_B = (double*)malloc(buffer_size);
    double *test_C = (double*)malloc(buffer_size);

    int index;

    // Fill test matrices with random values
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            index = i*N + j;
            test_A[index] = (double)rand()/(double)(RAND_MAX/max_value);
            test_B[index] = (double)rand()/(double)(RAND_MAX/max_value);
            test_C[index] = 0.0;

        }
    }

    // Fill all individual matrices with same random values used in test matrices
    for(int b=0; b<batch_count; b++){
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){

                index = i*N + j;
                (A[b])[index] = test_A[index];
                (B[b])[index] = test_B[index];
                (C[b])[index] = test_C[index];

            }
        }
    }

    // Pass host matrix buffers (containing matrix values) to device buffers
    for(int b=0; b<batch_count; b++){

        cudaErrorCheck( cudaMemcpy(h_d_A[b], A[b], buffer_size, cudaMemcpyHostToDevice) );
        cudaErrorCheck( cudaMemcpy(h_d_B[b], B[b], buffer_size, cudaMemcpyHostToDevice) );
        cudaErrorCheck( cudaMemcpy(h_d_C[b], C[b], buffer_size, cudaMemcpyHostToDevice) );

    }

    double alpha = 1.0;
    double beta  = 0.0;

    // Call host dgemm to get expected values
    dgemm("n", "n", N, N, N, alpha, test_A, N, test_B, N, beta, test_C, N);

    cublasStatus_t cublas_stat;
    cublasHandle_t handle;
    cublas_stat = cublasCreate(&handle);

    if(cublas_stat != CUBLAS_STATUS_SUCCESS){
        printf("cublasCreate failed with code %d\n", cublas_stat);
        return EXIT_FAILURE;
    }

    // Call batched device dgemm
    cublas_stat = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N, batch_count);

    if(cublas_stat != CUBLAS_STATUS_SUCCESS){
        printf("cublasDgemmBatched failed with code %d\n", cublas_stat);
        return EXIT_FAILURE;
    }

    // Copy results back from device to host
    for(int b=0; b<batch_count; b++){
        cudaErrorCheck( cudaMemcpy(C[b], h_d_C[b], buffer_size, cudaMemcpyDeviceToHost) );
    }

    // Make sure all matrices have the same results as the host dgemm test
    for(int b=0; b<batch_count; b++){
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){

                index = i*N + j;
                if( (C[b])[index] != test_C[index] ){
                    printf("Error - (C[%d])[%d] = %.3f instead of %.3f. Exiting...\n", b, index, (C[b])[index], test_C[index]);
                    exit(1);

                }
            }
        }
    }   

    cublas_stat = cublasDestroy(handle);

    if(cublas_stat != CUBLAS_STATUS_SUCCESS){
        printf("cublasDgemmBatched failed with code %d\n", cublas_stat);
        return EXIT_FAILURE;
    }

    // Clean up memory
    for(int b=0; b<batch_count; b++){
        cudaErrorCheck( cudaFree(h_d_A[b]) );
        cudaErrorCheck( cudaFree(h_d_B[b]) );
        cudaErrorCheck( cudaFree(h_d_C[b]) );
        free(A[b]);
        free(B[b]);
        free(C[b]);
    }

    cudaErrorCheck( cudaFree(d_A) );
    cudaErrorCheck( cudaFree(d_B) );
    cudaErrorCheck( cudaFree(d_C) );
    free(h_d_A);
    free(h_d_B);
    free(h_d_C);
    free(A);
    free(B);
    free(C);

    printf("__SUCCESS__\n");    

    return 0;
}
