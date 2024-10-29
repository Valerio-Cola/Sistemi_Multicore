#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Deve ritornare un vettore di interi
int* create_random_vector(int len) {
    // Alloco spazio per un vettore di interi
    int* vect = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
        vect[i] = rand() % 100;
    }
    return vect;
}

void print_vettore(int* vect, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", vect[i]);
    }
    printf("\n");
}
/*
    L'obiettivo è di sommare due vettori a b di interi generati randomicamente

    Ogni processo riceverà mediante scatter un pezzo di entrambi i vettori,
    li sommerà e li invierà a un vettore finale mediante gather, funzione che permette
    di concatenare i sottovettori di tutti i processi in uno unico

    Es: 2 vettori lunghi 10
        2 processi ognuno riceve 2 sottovettori lunghi 5
        
        Root genera:
        Vettore a = 20 21 22 23 24 25 26 27 28 29  
        Vettore b = 10 11 12 13 14 15 16 17 18 19

        Root ottiene:
        Sottovettore a = 20 21 22 23 24
        Sottovettore b = 10 11 12 13 14

        Rank 1 ottiene:
        Sottovettore a = 25 26 27 28 29 
        Sottovettore b = 15 16 17 18 19

        Root somma in c e invia in vettore finale:
        Vettore c = 30 32 34 36 38

        Rank 1 somma in c e invia in vettore finale:
        Vettore c = 40 42 44 46 48

        Root stampa finale:
        Vettore finale = 30 32 34 36 38 40 42 44 46 48

*/
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *a, *b;
    // Primo elemento dopo nome programma
    // Inserisco da comando la lunghezza del vettore
    int len_vect = atoi(argv[1]);

    // Verifico se il vettore può essere interamente diviso in parti uguali
    if (len_vect % size != 0) {
        printf("Il vettore deve essere diviso tra i processi in modo uniforme\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Se esegue root
    if (rank == 0) {
        // Creo i vettori con numeri randomici con funzione ausiliaria
        a = create_random_vector(len_vect);
        b = create_random_vector(len_vect);

        // Stampo vettore con funzione ausiliaria
        printf("Vettore completo a = ");
        print_vettore(a, len_vect);

        printf("Vettore completo b = ");
        print_vettore(b, len_vect);

        // Invia il vettore spezzato, con MPI_IN_PLACE anche root riceve i sottovettori
        MPI_Scatter(a, len_vect / size, MPI_INT, MPI_IN_PLACE, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, len_vect / size, MPI_INT, MPI_IN_PLACE, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
    
    } else {
        // Creo array in cui ricevere parte del vettore
        a = (int*)malloc(len_vect / size * sizeof(int));
        b = (int*)malloc(len_vect / size * sizeof(int));

        // Riceve vettore spezzato
        MPI_Scatter(NULL, len_vect / size, MPI_INT, a, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, len_vect / size, MPI_INT, b, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Stampo le varie parti assegnate a ogni processo
    printf("Sottovettore di a assegnato a rank %d = ", rank);
    print_vettore(a, len_vect / size);

    printf("Sottovettore di b assegnato a rank %d = ", rank);
    print_vettore(b, len_vect / size);

    // Vettore su cui ogni processo effettua somma
    int* c = (int*)malloc(len_vect / size * sizeof(int));

    // Ogni processo somma i valori dei propri sottovettori di a,b
    for (int i = 0; i < len_vect / size; i++) {
        c[i] = a[i] + b[i];
    }

    // Processo root genera il vettore finale
    int* finale = NULL;
    if (rank == 0) {
        finale = (int*)malloc(len_vect * sizeof(int));
    }

    // Ogni processo invia la somma c al vettore finale
    MPI_Gather(c, len_vect / size, MPI_INT, finale, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Di nuovo il root stampa il vettore finale
    if (rank == 0) {
        printf("Vettore finale: ");
        print_vettore(finale, len_vect);
        free(finale);
    }

    // Con malloc bisogna liberare memoria
    free(a);
    free(b);
    free(c);

    MPI_Finalize();
    return 0;
}
