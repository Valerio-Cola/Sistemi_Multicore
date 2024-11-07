#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>

/*
Un blocco di codice è detto THREAD-SAFE se può essere eseguito da più thread contemporaneamente senza incorrere in problemi di sincronizzazione.
Supponiamo di voler "tokenizzare" un file di testo. I tokens sono le parole separate da spazi, tabulazioni e newline.

Approccio semplice: dividere il file in righe e assegnare le riche ai thread con turnazione round-robin.
La prima riga va al thread 0, la seconda al thread 1, ... , la riga t va al thread t, la riga t+1 al thread 0, ecc.
Possiamo serializzare l'accesso alle righe usando semafori
*/

//Dopo che un thread ha letto una riga, la tokenizza con la funzione strtok
char* strtok(char* string, const char* delim);
//la funzione strtok è THREAD-UNSAFE, quindi non possiamo chiamarla contemporaneamente da più thread

/*
In C esistono anche altre funzioni di libreria THREAD-UNSAFE, come random in stdlib.h o localtime in time.h
in alcuni casi, la libreria standard di C fornisce versioni THREAD-SAFE di queste funzioni, dette funzioni "rientranti" (reentrant)
come strtok_r, random_r, localtime_r.
La differenza è che la cache che viene utilizzata da queste funzioni glie la forniamo noi come argomento,
in questo modo ogni thread può usare una propria cache.
*/

// Tokenizzatore multi-thread THREAD-UNSAFE
// Definisci la lunghezza massima per una riga
#define MAX 1024

// Definisci il numero di thread
int thread_count = 4; // Puoi cambiare questo valore secondo necessità

// Definisci l'array di semafori
sem_t semaphores[4]; // Regola la dimensione in base al numero di thread

void* Tokenize(void* rank) {
    long my_rank = (long) rank;
    int count;
    int next = (my_rank + 1) % thread_count;
    char *fg_rv;
    char my_line[MAX];
    char *my_string;

    // Aspetto il mio turno
    Sem_wait(&semaphores[my_rank]);
    // Leggo la mia riga
    fg_rv = fgets(my_line, MAX, stdin);
    // Passo la riga al prossimo thread
    Sem_post(&semaphores[next]);
    // Tokenizzo la riga
    while (fg_rv != NULL) {
        printf("Thread %ld > my line = %s\n", my_rank, my_line);
        count = 0;
        
        // separa la linea quando incontra \t o \n
        my_string = strtok(my_line, " \t\n"); 
        while (my_string != NULL) {
            count++;
            printf("Thread %ld > token %d = %s\n", my_rank, count, my_string);
         
            // continua a tokenizzare la stringa
            my_string = strtok(NULL, " \t\n");  
        }

        Sem_wait(&semaphores[my_rank]);
        fg_rv = fgets(my_line, MAX, stdin);
        Sem_post(&semaphores[next]);
    }

    return NULL;
} // se eseguiamo questa funzione su più thread avremo dei problemi

/*
Le call di MPI sono thread-safe? NO, meglio inizializzare con:
        int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
    uguale per i mutex, cond e lock:
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
        pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;

    required e provided sono i livelli di threading in input e output:
        - MPI_THREAD_SINGLE equivalente ad utilizzare MPI_Init
        - MPI_THREAD_FUNNELED Solo il thread principale può utilizzare le call MPI
        - MPI_THREAD_SERIALIZED solo un thread alla volta può fare chiamate MPI, meglio utilizzare
            mutex per assicurare tale condizione
        - MPI_THREAD_MULTIPLE qualsiasi thread può fare chiamate MPI e la libreria si assicura che
            i vari accessi vengano fatti in modo sicuro. Poco efficiene
*/