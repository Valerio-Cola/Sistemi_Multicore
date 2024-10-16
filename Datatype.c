/* Anche qui bella porcoddio che speedrunnata, da finire

I "derived datatypes" sono usati per rappresentare insiemi di dati in memoria anche di diverso tipo, 
memorizzandone i tipi e le posizioni relative. Questo permette a una funzione che invia dati di raccogliere 
gli elementi corretti dalla memoria prima di spedirli, e a una funzione che riceve dati di distribuirli
correttamente in memoria quando vengono ricevuti.

Esempi

    1)
    struct t{
        double a;
        double b;
        int n;
    }

*/

#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
/*
    Creazione del derived datatype 
        struct t{
            double a;
            double b;
            int n;
        }
*/
    MPI_Datatype t;

    int block_lengths[3] = {1, 1, 1}; // Numero di elementi per tipo
    MPI_Aint displacements[3] = {0,16,24}; // Offset di memoria
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

    // Genera struttura
    MPI_Type_create_struct(3, block_lengths, displacements, types, &t);
    
    // Necessario per essere ottimizzato per le comunicazioni
    MPI_Type_commit(&t);

        /*   code   */

    // Libera memoria
    MPI_Type_free(&t);

    MPI_Finalize();
    return 0;
}
