#include <cuda.h>
#include <stdio.h>

#define RADIUS 3
#define BLOCK_SIZE 256

/*
    Problema, i thread lavorano in parallelo, in caso ci fossero più di 32 elementi vuol dire che verranno utilizzti più warp (ognuno 32)
    e quindi gli elementi caricati e calcolati dal primo potrebbero non essere soncronizzati con gli altri warp
    Soluzione: __syncthread() è una barriera che verifica che tutti i thread del blocco abbiano completato il calcolo prima di continuare
*/
__global__ void stencil_1D(int *in, int *out) {
    // crea un array in memoria condivisa per memorizzare i dati del blocco
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    
    // Calcola l'indice globale e locale
    int gindex = blockDim.x * blockIdx.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;
    
    // Legge gil elementi in memoria globale e li copia in memoria condivisa
    temp[lindex] = in[gindex];
    
    if (threadIdx.x < RADIUS) {
        // I primi 3 elementi e gli ultimi 3 devono caricare in memoria condivisa anche elementi della regione di halo
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    
    // Sincronizza i thread del blocco in modo che tutti abbiano copiato i dati in memoria condivisa prima di continuare
    __syncthreads();

    // Ogni thread calcola la somma dei 7 elementi (3 a sinistra, 3 a destra e se stesso)
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        result += temp[lindex + offset];

    // Scrive il risultato in memoria globale
    out[gindex] = result;
}

int main(int argc, char const *argv[])
{
    return 0;
}
