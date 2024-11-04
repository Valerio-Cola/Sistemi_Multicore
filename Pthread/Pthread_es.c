#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// La variabile è globale così che venga utilizzata sia dal main che dai thread 
// Altrimenti dovrei passare alla funzione la variabile
int thread_count;

void *Hello(void* rank){
    long my_rank = (long) rank;  // Cast del puntatore a void in long per sistemi a 64-bit
    printf("Hello from thread %ld of %d\n", my_rank, thread_count);
    return NULL;
} 

int main(int argc, char const *argv[]){

    long thread;

    pthread_t* thread_handles;

    // Converte stringa in intero, prende il numero di thread da riga comando
    thread_count = strtol(argv[1], NULL, 10);
    
    // Allocazione di memoria per i thread numero di thread * dimensione di un puntatore a thread
    // è la struttura da passare a create e join
    thread_handles = malloc(thread_count*sizeof(pthread_t));

    // Per ogni thread, partendo dal thread 0
    for (thread = 0; thread < thread_count; thread++){
        pthread_create(&thread_handles[thread], NULL, Hello, (void*) thread);
    }

    printf("Hello from the main thread\n");

    // Ogni thread attende la fine del predecessore
    for(thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);
    return 0;
}
