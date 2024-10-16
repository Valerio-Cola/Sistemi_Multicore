#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// Deve ritornare un vettore di interi
int* create_random_vector(int len){
    
    // Alloco spazio per un vettore di interi
    int* vect = (int*)malloc(len*sizeof(int));
    
    for(int i = 0; i < len; i++){
        vect[i] = rand() % 100;
    }
    return vect;
}


void print_vettore(int* vect, int n){
    for(int i = 0; i<n; i++){
        printf("%d ", vect[i]);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *a, *b;

    //Primo elemento dopo nome programma
    //Inserisco da comando la lunghezza del vettore
    int len_vect = atoi(argv[1]);

    //Verifico se il vettore può essere interamente diviso in parti uguali
    if( len_vect % size != 0){
        printf("Il vettore deve essere diviso tra i processi in modo uniforme\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Se esegue root 
    if(rank == 0){

        // Creo i vettori con numeri randomici con funzione ausiliaria
        a = create_random_vector(len_vect);
        b = create_random_vector(len_vect);

        // Stampo vettore con funzione ausiliaria
        printf("Vettore completo a = ");
        print_vettore(a, len_vect);

        // Invia il vettore spezzato
        // Array da inviare, lunghezza dei segmenti, tipo dato, buffer di appoggio non deve ricevere nulla, lunghezza da inviare, tipo dato, comunicatore
        MPI_Scatter(a, len_vect/size, MPI_INT, MPI_IN_PLACE, len_vect/size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, len_vect/size, MPI_INT, MPI_IN_PLACE, len_vect/size, MPI_INT, 0, MPI_COMM_WORLD);

    }else{

        // Se no creo array in cui ricevere parte del vettore
        a = (int*)malloc(len_vect/size*sizeof(int));
        b = (int*)malloc(len_vect/size*sizeof(int));

        // Riceve vettore spezzato
        // NULL perchè non deve inviare nulla
        MPI_Scatter(NULL, len_vect/size, MPI_INT, a, len_vect/size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, len_vect/size, MPI_INT, b, len_vect/size, MPI_INT, 0, MPI_COMM_WORLD);

    }
    // Stampo le varie parti assegnate a ogni processo
    printf("Sottovettore di a asssegnato a rank %d = ", rank);
    print_vettore(a, len_vect/size);

    //Somma dei pezzi con gather, da finire
 /*   int* c = (int*)malloc(n/size*sizeof(int));

    for(int i = 0; i < n/size; i++){
        c[i] = a[i] + 
    }
    int* c_finale = NULL;
    
    MPI_Gather();
*/

    // Con malloc bisogna liberare memoria
    free(a);
    free(b);

    MPI_Finalize();
    return 0;

}
