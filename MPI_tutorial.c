#include <stdio.h>
#include <mpi.h>
CAZZI
//Compilare
//mpicc hello_world_0.c -o hello_world_0

//Eseguire (-n è il numero di processi in parallelo)
//mpirun -Wall --oversubscribe -n 10 hello_world_0
//mpiexec -n 4 ./hello_world_0


int main(void) {
	int r = MPI_Init(NULL, NULL);

	
	//Dimensione e rank del comunicatore = collection of processes that can send messages to each other
	//Ogni core ha rank univoco
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(r != MPI_SUCCESS) {
		printf("errore\n");
		MPI_Abort(MPI_COMM_WORLD, r);
	} 
		
	/*
	
	-La print stampa non in ordine di rank poichè l'ordine di esecuzione dei processi è casuale
	
		printf("hello, world from process %d out of %d\n", rank, size);
	
	
	-Struttura vdei messaggi
	
	int MPI_Send(
			
			void* msg_buf_p, 		   	Puntatore ai dati
			int msg_size,  				Numero di elementi nel messaggio
			MPI_Datatype msg_type, 	   Tipo di dato nel messaggio
			int dest,
			int tag,
			MPI_Comm communicator
	)
	
	int MPI_Recv(
			
			void* msg_buf_p,  			Puntatore ai dati
			int buf_size,					
			MPI_Datatype buf_type,
			int source,
			int tag,
			MPI_Comm communicator,
			MPI_Status* status_p
	)
	
	*/
	
	
	//Voglio che il processo con rank 0 prenda le stringhe da tutti gli altri rank e le stampi in terminale in ordine 
		
	//NOTA che ogni processo ha il proprio spazio di memoria, i tipi di dato sono equivalenti a quelli di c: MPI_Char, MPI_Int, MPI_Float....
	if(rank == 0){
			//Quando è rank 0 stampa la propria stringa e per ogni rank maggiore di 0 riceve il messaggio e lo stampa
			printf("hello, world from process %d out of %d\n", rank, size);
	
			//NOTA I messaggi vengono ricevuti in ordine casuale, ma è possibile ordinarli
			
			
			/*  TAG:
					MPI_ANY_SOURCE è possibile ricevere da chiunque 
					MPI_ANY_TAG è possibile ricevere msg con qualsiasi tag
			*/
			int i;
			for(i=1; i < size; i++){
					char str[256];
					
					//Riceve una stringa str con 256 elementi del tipo CHAR da inviare al processo con Rank i, tag nullo = 0, comunicatore MPI_COMM_WORLD e ne ignora lo stato.
					MPI_Recv(str, 256, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					printf("%s", str);
			}
			
	}else{
			//Per tutti gli altri rank
			char str[256];
			sprintf(str, "hello, world from process %d out of %d\n", rank, size);	
			
			//Invia la stringa str con 256 elementi del tipo CHAR da inviare al processo con Rank 0, tag nullo = 0 e comunicatore MPI_COMM_WORLD
			MPI_Send(str, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
		
	}
	
	/*
		Si puo ricevere msg senza inserire direttamente la dimensione del buffer. è possibile calcolarlo con questa funzione mediante lo status
		
		int MPI_Get_count(
			MPI_Status status_p,
			MPI_Datatype type
		)
	*/
	
	MPI_Finalize();
	return 0;
}








