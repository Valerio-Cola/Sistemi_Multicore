#include <mpi.h>
#include <stdio.h>
/* Permette a tutti i processi di scambiare dati con tutti gli altri processi
    
    Supponiamo che ci siano 4 processi (0, 1, 2, e 3). Ognuno ha un buffer di invio (sendbuf) e uno di ricezione (recvbuf).
        Process 0 invia: [0, 1, 2, 3]
        Process 1 invia: [100, 101, 102, 103]
        Process 2 invia: [200, 201, 202, 203]
        Process 3 invia: [300, 301, 302, 303]

    Ora, MPI_Alltoall scambia questi dati tra tutti i processi. Ecco cosa riceve ogni processo:
        Process 0 riceve: [0, 100, 200, 300]
        Process 1 riceve: [1, 101, 201, 301]
        Process 2 riceve: [2, 102, 202, 302]
        Process 3 riceve: [3, 103, 203, 303]

*/
int main(int argc, char** argv) {
    // Inizializza l'ambiente MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Ottieni il rank del processo corrente
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Ottieni il numero totale di processi
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Buffer di invio e ricezione 
    int sendbuf[size]; 
    int recvbuf[size]; 

    // Inizializza il buffer di invio con valori unici per ciascun processo
    for (int i = 0; i < size; i++) {
        sendbuf[i] = rank * 100 + i;
    }

    // Esegui Alltoall per scambiare i dati tra tutti i processi
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    // Stampa il buffer ricevuto per ciascun processo
    printf("Process %d ha ricevuto:", rank);
    for (int i = 0; i < size; i++) {
        printf(" %d", recvbuf[i]);
    }
    printf("\n");

    // Finalizza l'ambiente MPI
    MPI_Finalize();
    return 0;
}