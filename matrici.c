/* Moltiplicare una matrice per un vettore

    Data una matrice n*n A[riga,colonna] e un vettore V con n elementi 
    Il vettore risultante R sarà: R[0] =  A[0,1] * V[1] + A[0,2] * V[2] + ...
                                  R[1] =  A[1,1] * V[1] + A[1,2] * V[2] + ... 
                                  Fino a R[n]

    Nota che una matrice viene inserita in memoria come un vettore unico quindi sarà del tipo:
        Vettore_Matrice = Riga1 + Riga2 + ... 

    Ovviamente le righe della matrice devono essere divise e senza resti tra i processi
    Se la matrice è da 4 righe potrò utilizare 1,2,4 processi
    Il numero di colonne non è vincolato quindi vanno bene matrici non quadrate

*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Semplici funzioni di stampa (banali si fanno all'asilo)
void print_vector(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}
void print_matrix(int* matrix, int righe, int colonne) {
    for (int i = 0; i < righe; i++) {
        for (int j = 0; j < colonne; j++) {
            printf("%d ", matrix[i * colonne + j]);
        }
        printf("\n");
    }
}

void matrix_vector_multiplication_mpi(int* A, int* x, int* finale, int righe, int colonne, int rank, int size) {
    
    // Righe assegnate a ogni processo
    int righe_locali = righe / size;

    // Allocazione della sottomatrice e vettore in cui inserire risultati della moltiplicazione
    int* local_A = (int*)malloc(righe_locali * colonne * sizeof(int));
    int* finale_locale = (int*)malloc(righe_locali * sizeof(int));

    // Riceve il pezzo di matrice inviando tot righe a ogni processo
    MPI_Scatter(A, righe_locali * colonne, MPI_INT, local_A, righe_locali * colonne, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast del vettore x a tutti i processi, ogni processo deve eseguirlo per garantire che tutti abbiano i dati necessari per il calcolo.
    MPI_Bcast(x, colonne, MPI_INT, 0, MPI_COMM_WORLD);

    // Il processo calcola la moltiplicazione per ogni riga assegnata
    for (int i = 0; i < righe_locali; i++) {
        // Inizializza risultato ogni volta che passa a una nuova riga
        finale_locale[i] = 0;

        // Scorre la riga della matrice, per ogni elemento lo moltiplica con il corrispondente nel vettore x 
        for (int j = 0; j < colonne; j++) {
            finale_locale[i] += local_A[i * colonne + j] * x[j];
        }
    }

    // Invio a finale il vettore con i risultati di ogni processo, il processo invia un numero per ogni riga assegnata
    MPI_Gather(finale_locale, righe_locali, MPI_INT, finale, righe_locali, MPI_INT, 0, MPI_COMM_WORLD);

    free(local_A);
    free(finale_locale);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // In input da terminale, il numero di colonne equivale alla lunghezza del vettore
    int righe = atoi(argv[1]); 
    int colonne = atoi(argv[2]); 
    
    // Verifico che le righe possano essere distribuite correttamente
    if (righe % size != 0) {
        printf("Le righe devono essere divise tra i processi in modo uniforme\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    srand(time(NULL));

    // Vettore finale
    int* finale = (int*)malloc(righe * sizeof(int));

    // Generazione e allocazione matrice e vettore da moltiplicare
    int* A = (int*)malloc(righe * colonne * sizeof(int));
    int* x = (int*)malloc(colonne * sizeof(int));
    if (rank == 0) {

        // Genero matrice
        for (int i = 0; i < righe; i++) {
            for (int j = 0; j < colonne; j++) {
                A[i * colonne + j] = rand() % 20;;
            }
        }

        // Genero vettore
        for (int i = 0; i < colonne; i++) {
            x[i] = rand() % 20;
        }
        
        printf("Matrice A:\n");
        print_matrix(A, righe, colonne);
        
        printf("Vettore x:\n");
        print_vector(x, colonne);
    }

    matrix_vector_multiplication_mpi(A, x, finale, righe, colonne, rank, size);

    if (rank == 0) {
        printf("Resulting vector finale:\n");
        print_vector(finale, righe);
        free(A);
    }

    free(x);
    free(finale);
    MPI_Finalize();
    return 0;

}