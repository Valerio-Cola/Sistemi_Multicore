#include <stdio.h>
#include <mpi.h>

int main(int argc, char const *argv[])
{
    int r = MPI_Init(NULL, NULL);

	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int send_r = 19;
    int send_l = 23;

    int recv_l, recv_r; 

    /* Scambio msg utilizzando Isend Irecv che non blocca mai
       Ssend e Srecv sono bloccanti
       Bsend e Brecv sono bufferizzati
    */

    //Utilizzo array per le richieste
    MPI_Request lista_richieste[4];

    //Send a destra e sinistra 
    //  Puntatore a dato da inviare, Tipo, A chi mandarlo, tag, comunicatore, puntatore alla richiesta
    MPI_Isend(&send_r, 1, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD, &lista_richieste[0]);  
    MPI_Isend(&send_l, 1, MPI_INT, (rank-1)%size, 0, MPI_COMM_WORLD, &lista_richieste[1]);
    
    //Receive 
    MPI_Irecv(&recv_r, 1, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD, &lista_richieste[2]);
    MPI_Irecv(&recv_l, 1, MPI_INT, (rank-1)%size, 0, MPI_COMM_WORLD, &lista_richieste[3]);

    //Attendo che tutte le richieste vengano completate ignorandone lo stato
    MPI_Waitall(4, lista_richieste, MPI_STATUS_IGNORE);



    //In questo caso inizializzo ogni singola richiesta
    MPI_Request request_send_r, request_send_l;
    MPI_Request request_recv_r, request_recv_l;

    //Send  
    MPI_Isend(&send_r, 1, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD, &request_send_r);
    MPI_Isend(&send_l, 1, MPI_INT, (rank-1)%size, 0, MPI_COMM_WORLD, &request_send_l);
    
    //Receive
    MPI_Irecv(&recv_r, 1, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD, &request_recv_r);
    MPI_Irecv(&recv_l, 1, MPI_INT, (rank-1)%size, 0, MPI_COMM_WORLD, &request_recv_l);

    //Per ogni richiesta attendo che si concluda 
    MPI_Wait(&request_send_l, MPI_STATUS_IGNORE);
    MPI_Wait(&request_send_r, MPI_STATUS_IGNORE);
    MPI_Wait(&request_recv_l, MPI_STATUS_IGNORE);
    MPI_Wait(&request_recv_r, MPI_STATUS_IGNORE);


    MPI_Finalize();
    return 0;
}
