#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int totale = 0;
    int nel_cerchio = 0;
    int numero_lanci_per_processo = 10;

    // Inizializzazione di MPI
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Sommare il proprio rank permette di generare numeri casuali per ogni processo
    srand(time(NULL)+rank);

    // Ogni processo esegue il suo calcolo
    for (int i = 0; i < numero_lanci_per_processo; i++) {
        double x = (double)rand() / RAND_MAX * 2 - 1;
        double y = (double)rand() / RAND_MAX * 2 - 1;
        if (x*x + y*y <= 1) {
            nel_cerchio++;
        }

    }

    // Se il punto è nel cerchio verrà inviata nel_cerchio da sommare a totale
    MPI_Reduce(&nel_cerchio, &totale, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Il processo root calcola π
    if (rank == 0) {
        // 4*totale punti nel cerchio / punti per processo*numero processi 
        double pi = 4*(double)totale/((double)numero_lanci_per_processo*size);
        printf("Stima di pi greco: %f\n", pi);
    }

    // Finalizzazione di MPI
    MPI_Finalize();
    return 0;
}