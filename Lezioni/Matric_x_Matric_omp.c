#include <omp.h>
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

// Funzione per la moltiplicazione di matrici usando OpenMP
void matrix_multiplication_omp(int* A, int* B, int* C, int righe1, int colonne1, int righe2, int colonne2) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < righe1; i++) {
        for (int j = 0; j < colonne2; j++) {
            C[i * colonne2 + j] = 0;
            for (int k = 0; k < colonne1; k++) {
                C[i * colonne2 + j] += A[i * colonne1 + k] * B[k * colonne2 + j];
            }
        }
    }
}

int main(int argc, char** argv) {

    // In input da terminale, il numero di colonne equivale alla lunghezza del vettore
    int righe1 = atoi(argv[1]); 
    int colonne1 = atoi(argv[2]); 

    int righe2 = atoi(argv[3]); 
    int colonne2 = atoi(argv[4]);

    // Verifica che le colonne della prima matrice siano uguali alle righe della seconda
    if(colonne1 != righe2){
        printf("Le colonne della prima matrice devono essere uguali alle righe della seconda\n");
        return 1;
    }

    // Inizializzazione del generatore di numeri casuali
    srand(time(NULL));

    // Generazione e allocazione matrici
    int* A = (int*)malloc(righe1 * colonne1 * sizeof(int));
    int* B = (int*)malloc(righe2 * colonne2 * sizeof(int));
    int* C_Prodotto = (int*)malloc(righe1 * colonne2 * sizeof(int));

    
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
    

    // Chiamata alla funzione di moltiplicazione delle matrici
    matrix_multiplication_omp(A, B, C_Prodotto, righe1, colonne1, righe2, colonne2);

    
    printf("Matrice Prodotto:\n");
    print_matrix(C_Prodotto, righe1, colonne2);
    
    
    // Libera la memoria allocata
    free(A);
    free(B);
    free(C_Prodotto);

    return 0;
}