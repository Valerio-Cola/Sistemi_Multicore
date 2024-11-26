#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void print_matrix(int* matrix, int righe, int colonne) {
    for (int i = 0; i < righe; i++) {
        for (int j = 0; j < colonne; j++) {
            printf("%d\t", matrix[i * colonne + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void sum_adjacent(int* A, int* B, int righe, int colonne) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < righe; i++) {
        for (int j = 0; j < colonne; j++) {
            int sum = 0;
            // Somma elementi adiacenti
            if (i > 0) sum += A[(i-1) * colonne + j];         // Nord
            if (i < righe-1) sum += A[(i+1) * colonne + j];   // Sud
            if (j > 0) sum += A[i * colonne + (j-1)];         // Ovest
            if (j < colonne-1) sum += A[i * colonne + (j+1)]; // Est
            B[i * colonne + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s <righe> <colonne> <iterazioni>\n", argv[0]);
        return 1;
    }

    int righe = atoi(argv[1]);
    int colonne = atoi(argv[2]);
    int iterazioni = atoi(argv[3]);

    // Alloca solo due matrici
    int* A = (int*)malloc(righe * colonne * sizeof(int));
    int* B = (int*)malloc(righe * colonne * sizeof(int));

    // Inizializza la matrice A con numeri casuali
    srand(time(NULL));
    for (int i = 0; i < righe * colonne; i++) {
        A[i] = rand() % 21 - 10; // Numeri tra -10 e 10
    }

    printf("Matrice iniziale:\n");
    print_matrix(A, righe, colonne);

    // Esegui le iterazioni
    for (int i = 1; iterazioni > 0; i++) {
        sum_adjacent(A, B, righe, colonne);
        
        // Copia B in A e azzera B
        for(int i = 0; i < righe * colonne; i++) {
            A[i] = B[i];
            B[i] = 0;
        }

        printf("Iterazione %d:\n", i);
        print_matrix(A, righe, colonne);
        iterazioni--;
    }

    // Libera la memoria
    free(A);
    free(B);
    return 0;
}