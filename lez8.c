#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>


int* create_random_vector(int n){
    int* vect = (int*)malloc(n*sizeof(int));
    for(int i = 0; i < n; i++){
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
    int n = atoi(argv[1]);

    if(n%size != 0){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(rank == 0){
        a = create_random_vector(n);
        b = create_random_vector(n);

        print_vettore(a, n);
        // Invia il vettore spezzato
        MPI_Scatter(a, n/size, MPI_INT, MPI_IN_PLACE, n/size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, n/size, MPI_INT, MPI_IN_PLACE, n/size, MPI_INT, 0, MPI_COMM_WORLD);

    }else{
        a = (int*)malloc(n/size*sizeof(int));
        b = (int*)malloc(n/size*sizeof(int));

        // Riceve vettore spezzato
        MPI_Scatter(NULL, n/size, MPI_INT, a, n/size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, n/size, MPI_INT, b, n/size, MPI_INT, 0, MPI_COMM_WORLD);

    }
    printf("rank %d: a = ", rank);
    print_vettore(a, n/size);

    //Somma dei pezzi
 /*   int* c = (int*)malloc(n/size*sizeof(int));

    for(int i = 0; i < n/size; i++){
        c[i] = a[i] + 
    }
    int* c_finale = NULL;
    
    MPI_Gather();
*/
    free(a);
    free(b);

    MPI_Finalize();
    return 0;

}
