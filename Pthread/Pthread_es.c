#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
//gcc -g -Wall -o Pthread_Es Pthread_es.c -lpthread
//./pthread num_thread

int thread_count = 10;
void *hello(void* rank);

void *hello(void* rank){
    long my_rank = (long) rank;  // Cast del puntatore a void in long per sistemi a 64-bit
    printf("Hello from thread %ld of %d\n", my_rank, thread_count);
    return NULL;
}


int main(int argc, char const *argv[]){

    long thread;

    pthread_t* thread_handles;

    // Converte stringa in intero
    thread_count = strtol(argv[1], NULL, 10);
    
    // Allocazione di memoria per i thread numero di thread * dimensione di un puntatore a thread
    thread_handles = malloc(thread_count*sizeof(pthread_t));

    for (thread = 0; thread < thread_count; thread++){

        pthread_create(&thread_handles[thread], NULL, hello, (void*) thread);
    }

    printf("Hello from the main thread\n");

    for(thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);
    return 0;
}
