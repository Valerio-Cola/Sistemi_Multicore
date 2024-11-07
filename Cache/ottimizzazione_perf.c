#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "my_timer.h"



// Compila con i simboli di debug
// MAX e ITER sono definiti da riga di comando con -D
// gcc -g -D MAX=100 -D ITER=5 -o lez12 lez12.c


// Esegui l'analisi delle prestazioni salvando i risultati in un file
// perf record ./lez12

// Visualizza i risultati
// perf report

// Esegui l'analisi delle prestazioni e mostra risultati su terminale
// perf stat ./lez12

/*

Livelli ottimizzazione per minimizzare il tempo di esecuzione
    -O0: Nessuna ottimizzazione
    -O1: Ottimizzazione base
    -O2: Ottimizzazione più aggressiva
    -O3: Ottimizzazione massima
Con O3 si verifica il DCE (Dead Code Elimination) 
che elimina il codice non utilizzato, in questo caso la matrice

Si attiva solo se y viene dichiarata nel main
  
A non viene utilizzata dopo averla costruita e quindi non viene eseguito il codice
Banalmente se si stampa la matrice il compilatore non può fare DCE
 
NOTA inoltre che la matrice viene letta più velocemente riga per riga e non colonna per colonna

*/


double A[MAX][MAX];
double x[MAX];
double y[MAX];   


int main(int argc, char** argv) {
    int i,j,iter;
    srand(time(NULL));

    // Inizializza il vettore lungo MAX e la matrice MAX x MAX
    for (i = 0; i < MAX; i++) {
        x[i] = (double) rand() / RAND_MAX; // Random number between 0 and 1
        y[i] = 0.0;
        for (j = 0; j < MAX; j++)
            A[i][j] = (double) rand() / RAND_MAX; // Random number between 0 and 1
    }

    // ITER è il numero di olte che verrà fatta la moltiplicazione 
    double total_time = 0.0;
    for(iter = 0; iter < ITER; iter++){
        double start, stop;
        GET_TIME(start);

        // Più veloce riga per riga che colonna per colonna
        for (i = 0; i < MAX; i++)        
            for (j = 0; j < MAX; j++) 
//      for (j = 0; j < MAX; j++)
//          for (i = 0; i < MAX; i++)        
                y[i] += A[i][j]*x[j];
        GET_TIME(stop);
        total_time += stop-start;
    }


    printf("Average runtime %f sec\n", total_time/ITER);
}