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

// Funzione per la moltiplicazione di matrici usando MPI
void matrix_multiplication_mpi(int* A, int* B, int* C, int righe1, int colonne1, int righe2, int colonne2, int rank, int size) {
    
    // Righe assegnate a ogni processo
    int righe_locali = righe1 / size;

    // Allocazione della sottomatrice e matrice risultato locale
    int* local_A = (int*)malloc(righe_locali * colonne1 * sizeof(int));
    int* local_C = (int*)malloc(righe_locali * colonne2 * sizeof(int));

    // Riceve il pezzo di matrice A inviando tot righe a ogni processo
    MPI_Scatter(A, righe_locali * colonne1, MPI_INT, local_A, righe_locali * colonne1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast della matrice B a tutti i processi
    MPI_Bcast(B, righe2 * colonne2, MPI_INT, 0, MPI_COMM_WORLD);

    // Il processo calcola la moltiplicazione per ogni riga assegnata
    for (int i = 0; i < righe_locali; i++) {
        for (int j = 0; j < colonne2; j++) {
            local_C[i * colonne2 + j] = 0;
            for (int k = 0; k < colonne1; k++) {
                local_C[i * colonne2 + j] += local_A[i * colonne1 + k] * B[k * colonne2 + j];
            }
        }
    }

    // Raccolta dei risultati in C
    MPI_Gather(local_C, righe_locali * colonne2, MPI_INT, C, righe_locali * colonne2, MPI_INT, 0, MPI_COMM_WORLD);

    // Libera la memoria allocata
    free(local_A);
    free(local_C);
}

int main(int argc, char** argv) {
    
    // Inizializzazione di MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // In input da terminale, il numero di colonne equivale alla lunghezza del vettore
    int righe1 = atoi(argv[1]); 
    int colonne1 = atoi(argv[2]); 

    int righe2 = atoi(argv[3]); 
    int colonne2 = atoi(argv[4]);

    // Verifica che le colonne della prima matrice siano uguali alle righe della seconda
    if(colonne1 != righe2){
        printf("Le colonne della prima matrice devono essere uguali alle righe della seconda\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Verifica che le righe possano essere distribuite correttamente
    if (righe1 % size != 0) {
        printf("Le righe devono essere divise tra i processi in modo uniforme\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    } 

    // Inizializzazione del generatore di numeri casuali
    srand(time(NULL));

    // Generazione e allocazione matrici
    int* A = (int*)malloc(righe1 * colonne1 * sizeof(int));
    int* B = (int*)malloc(righe2 * colonne2 * sizeof(int));
    int* C_Prodotto = (int*)malloc(righe1 * colonne2 * sizeof(int));

    if (rank == 0) {
        // Genero matrice A
        for (int i = 0; i < righe1; i++) {
            for (int j = 0; j < colonne1; j++) {
                A[i * colonne1 + j] = rand() % 20;;
            }
        }
        printf("Matrice A:\n");
        print_matrix(A, righe1, colonne1);
        
        // Genero matrice B
        for (int i = 0; i < righe2; i++) {
            for (int j = 0; j < colonne2; j++) {
                B[i * colonne2 + j] = rand() % 20;;
            }
        }
        printf("Matrice B:\n");
        print_matrix(B, righe2, colonne2);
    }

    // Chiamata alla funzione di moltiplicazione delle matrici
    matrix_multiplication_mpi(A, B, C_Prodotto, righe1, colonne1, righe2, colonne2, rank, size);

    if(rank == 0){
        printf("Matrice Prodotto:\n");
        print_matrix(C_Prodotto, righe1, colonne2);
    }
    
    // Libera la memoria allocata
    free(A);
    free(B);
    free(C_Prodotto);

    // Finalizzazione di MPI
    MPI_Finalize();
    return 0;
}
