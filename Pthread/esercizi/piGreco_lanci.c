#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

int thread_count;
int numero_lanci_per_thread;
int nel_cerchio = 0;
pthread_mutex_t mutex;

void *lanci(){

    int nel_cerchio_locale = 0;
    
    for (int i = 0; i < numero_lanci_per_thread; i++) {
    double x = (double)rand() / RAND_MAX * 2 - 1;
    double y = (double)rand() / RAND_MAX * 2 - 1;
        if (x*x + y*y <= 1) {
            nel_cerchio_locale++;
        }
    }

    pthread_mutex_lock(&mutex);
    nel_cerchio += nel_cerchio_locale;
    pthread_mutex_unlock(&mutex);

    return NULL;

}

int main(int argc, char** argv)
{
    
    long thread;
    pthread_t* thread_handles;

    thread_count = strtol(argv[1], NULL, 10);
    numero_lanci_per_thread = strtol(argv[2], NULL, 10);

    thread_handles = malloc(thread_count*sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);

    // Per ogni thread, partendo dal thread 0
    for (thread = 0; thread < thread_count; thread++){
        pthread_create(&thread_handles[thread], NULL, lanci, NULL);
    }
    
    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    int totale = thread_count * numero_lanci_per_thread;
    // Rimozione del mutex
    pthread_mutex_destroy(&mutex);

    printf("Pi: %f\n", 4*(double)nel_cerchio/((double)totale));
    free(thread_handles);

    return 0;
}
