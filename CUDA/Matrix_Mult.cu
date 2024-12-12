__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width) {
    // Calcola la posizione globale 
    int row = blockIdx.y + threadIdx.y * blockDim.y;
    int col = blockIdx.x + threadIdx.x * blockDim.x;

    // Controlla che il thread sia all'interno della matrice
    if(col < Width && row < Width) {
        float Pvalue = 0;

        // 
        for(int k = 0; k < Width; ++k) {
            // Due accessi a memoria globale e due operazioni
            // intensita aritmetica FLOP/operazione = 1 FLOP/Byte = 8
            Pvalue += Md[row * Width + k] * Nd[k * Width + col];
        }
        Pd[row * Width + col] = Pvalue;
    }
}

// Miglioriamo la moltiplicazione tra matrici con il tiling, sfruttiamo la memoria condivisa
// invece che gli accessi alla globale
#define TILE_WIDTH 2
__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width) {
    // Allocazione della memoria condivisa per i tile di Md e Nd
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // Coordinate blocco di thread
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    // Coordinate thread nel blocco
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Coordinate della matrice globale che il thread deve calcolare
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Man mano il thead carica il proprio calcolo in questa variabile locale
    float Pvalue = 0;

    // Itera su tutti i tile di Md e Nd e carica i dati in memoria condivisa
    // Ricorda per ogni tile di entrambe le matrici sta caricando l'elemento con posizione uguale al thread
    for(int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        Mds[ty][tx] = Md[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = Nd[(ph * TILE_WIDTH + ty) * Width + Col];

        // Sincronizza tutti i thread prima che inizino a calcolare
        __syncthreads();

        // Dopo aver caricato in shared tutti i dati del primo tile delle 2 matrici
        // Ogni thread calcola il prodotto tra la riga di Md e la colonna di Nd
        for(int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // Tutti i thread devono aver finito di calcolare prima di caricare il prossimo tile
        __syncthreads();

        // Finito il primo ciclo si passa ai tile in posizione 2 di entrambe le matrici
    }

    // Ogni thread scrive il proprio risultato in Pd
    Pd[Row * Width + Col] = Pvalue;
}