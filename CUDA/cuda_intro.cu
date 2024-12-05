#include <stdio.h>
#include <cuda.h>
/*
Compilazione:    nvcc -arch=sm_20 -o sample sample.cu
Esecuzione:      ./sample

Funzione che puo essere chiamata sia dall'host che GPU, ma deve necessariamente essere eseguita su GPU
No valori di ritorno, di base neache I/O
 __global__ indica che la funzione è eseguita su GPU
 __device__ indica che la funzione è eseguita su GPU ma può essere chiamata solo da altre funzioni eseguite su GPU
 __host__ indica che la funzione è eseguita su CPU */

__global__ void hello(){
    printf("Hello World from GPU!\n");
}

int main(){
    // Chiamata alla funzione hello in un singolo blocco con 10 thread, la chiamata è asincrona quindi bisogna sincronizzare la GPU = attendere che finisca l'esecuzione
    hello<<<1,10>>>();
    cudaDeviceSynchronize();
    
    //Bisogna combinare i valori di threadIdx e blockIdx per ottenere un id univoco per ogni thread, poichè due thread su blocchi diversi potrebbero avere stesso threadIdx
    int myID = (blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) 
                * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;

    // Numero GPU disponibili
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        printf("No CUDA device found\n");
    }else{
        // Per ogni device stampa il nome ma nella struttura cudaDeviceProp ci sono molte altre informazioni
        cudaDeviceProp prop;
        for(int i = 0; i < deviceCount; i++){
            cudaGetDeviceProperties(&prop, i);
            printf("Device %d: %s\n", i, prop.name);
        }
    }

    struct cudaDeviceProp {
        char name[256]; // Nome del device
        int major; // Major compute capability number
        int minor; // Minor compute capability number
        int maxGridSize[3]; // Dimensioni massime della griglia
        int maxThreadsDim[3]; // Dimensioni massime dei blocchi
        int maxThreadsPerBlock; // Numero massimo di thread per blocco
        int maxThreadsPerMultiProcessor; // Numero massimo di thread per SM
        int multiProcessorCount; // Numero di SM
        int regsPerBlock; // Numero di registri per blocco
        size_t sharedMemPerBlock; // Shared memory available per block in bytes
        size_t totalGlobalMem; // Global memory available on device in bytes
        int warpSize; // Warp size in threads
    };
    

    return 0;
}
