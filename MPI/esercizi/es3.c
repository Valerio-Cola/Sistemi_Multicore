#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Funzione per stampare una matrice
void print_matrix(int* matrix, int righe, int colonne) {
    for (int i = 0; i < righe; i++) {
        for (int j = 0; j < colonne; j++) {
            printf("%d ", matrix[i * colonne + j]);
        }
        printf("\n");
    }
}

// Funzione per calcolare la somma degli elementi adiacenti usando MPI
void sum_adjacent_mpi(int* A, int* B, int righe, int colonne, int rank, int size) {
    
    // Righe assegnate a ogni processo
    int righe_locali = righe / size;

    // Allocazione della sottomatrice locale
    int* local_A = (int*)malloc(righe_locali * colonne * sizeof(int));
    int* local_B = (int*)malloc(righe_locali * colonne * sizeof(int));

    // Riceve il pezzo di matrice A inviando tot righe a ogni processo
    MPI_Scatter(A, righe_locali * colonne, MPI_INT, local_A, righe_locali * colonne, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcolo della somma degli elementi adiacenti
    for (int i = 0; i < righe_locali; i++) {
        for (int j = 0; j < colonne; j++) {
            int sum = 0;
            if (i > 0) sum += local_A[(i - 1) * colonne + j]; // sopra
            if (i < righe_locali - 1) sum += local_A[(i + 1) * colonne + j]; // sotto
            if (j > 0) sum += local_A[i * colonne + (j - 1)]; // sinistra
            if (j < colonne - 1) sum += local_A[i * colonne + (j + 1)]; // destra
            local_B[i * colonne + j] = sum;
        }
    }

    // Raccolta dei risultati in B
    MPI_Gather(local_B, righe_locali * colonne, MPI_INT, B, righe_locali * colonne, MPI_INT, 0, MPI_COMM_WORLD);

    // Libera la memoria allocata
    free(local_A);
    free(local_B);
}

int main(int argc, char** argv) {
    
    // In input da terminale, il numero di colonne equivale alla lunghezza del vettore
    int righe = atoi(argv[1]); 
    int colonne = atoi(argv[2]); 

    int iterazioni = atoi(argv[3]); 
    // Inizializzazione del generatore di numeri casuali
    srand(time(NULL));

    // Generazione e allocazione matrici
    int* A = (int*)malloc(righe * colonne * sizeof(int));

    // Array di appoggio
    int* B = (int*)malloc(righe * colonne * sizeof(int));
    
    // Genero matrice A
    for (int i = 0; i < righe; i++) {
        for (int j = 0; j < colonne; j++) {
            A[i * colonne + j] = rand() % 20;
        }
    }

    // Inizializzazione di MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank == 0){

        printf("Matrice A:\n");
        print_matrix(A, righe, colonne);
    }

    // Verifica che le righe possano essere distribuite correttamente
    if (righe % size != 0) {
        printf("Le righe devono essere divise tra i processi in modo uniforme\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 1; iterazioni > 0; i++) {
        sum_adjacent_mpi(A, B, righe, colonne, rank, size);
        
        // Copia B in A e azzera B
        for(int i = 0; i < righe * colonne; i++) {
            A[i] = B[i];
            B[i] = 0;
        }

        // Stampa la matrice aggiornata
        if (rank == 0) {
            printf("Matrice dopo iterazione %d:\n", i);
            print_matrix(A, righe, colonne);
        }

        iterazioni--;
    }

    // Finalizzazione di MPI
    MPI_Finalize();
        
    // Libera la memoria allocata
    free(A);
    free(B);
    return 0;
}