#include <cuda.h>
#include <stdio.h>

// Macro utilizzata per verificare che le chiamate a funzione di CUDA non diano errore 
#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;          \
    if (_m_cudaStat != cudaSuccess) {         \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1);                              \
    }                                         \
}

// Kernel function to perform element-wise vector multiplication
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    // Fa lavorare solo i thread che hanno un indice minore della dimensione del vettore
    // Poichè in questo caso abbiamo il blocco 2 in cui lavora solo un thread
    // Tutti gli altri non devono lavorare poichè si rischia di accedere a zone di memoria non allocate
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}

// Funzione che esegue il prodotto di due vettori 
// h_A e h_B sono i due vettori da moltiplicare allocati nell'host
// h_C è il vettore risultante allocato nell'host
// n è la dimensione dei vettori
void vect_x_vect(float* h_A, float* h_B, float* h_C, int n){
    // Dimensione dei vettori
    int size = n * sizeof(float);
    
    // Dati allocati su GPU/device
    float *d_A, *d_B, *d_C;

    // Allocazione memoria per i due vettori A B su GPU
    // Prendono in input l'indirizzo del puntatore e la dimensione in byte
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    // Copia dei dati all'interno dei vettori da host a device
    // Indirizzo destinazione e sorgente, dimensione in byte, tipo di copia
    // cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault
    // NOTA: cudaMemcpy è una funzione sincrona, quindi blocca il thread finchè la copia non è completata
    //       I puntatori h_ d_ allocati quindi sono diversi tra GPU e CPU, ma con CUDA 6 è stata introdotta la Unified Memory che permette di avere un unico spazio di memoria
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Allocazione memoria per il vettore risultante C su GPU
    cudaMalloc((void**)&d_C, size);

    // Chiamata al kernel per eseguire la moltiplicazione dei due vettori
    // quindi ogni blocco ha 256 thread e ci sono ceil(n/256.0) blocchi
    // Se array è da 257 elementi partiranno 2 blocchi da 256 thread, nel primo lavorano tutti 
    // Nel secondo solo 1 thread 
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    // Copia dei dati all'interno del vettore risultante da GPU a host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main(int argc, char const *argv[])
{
    
    return 0;
}
