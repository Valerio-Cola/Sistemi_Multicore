#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <semaphore.h>

int thread_count;
int n;
double sum;
int flag = 0;

// Creazione mutex
// Per utilizzare lock e unlock
pthread_mutex_t mutex;

void *piGreco(void *rank){

    long my_rank = (long) rank;
    int i;
    int local_n = n / thread_count;
    int posiz_iniziale = my_rank * local_n;
    int posiz_finale = (my_rank + 1) * local_n - 1;
    
    double my_sum = 0.0;
    double factor;
    if(posiz_iniziale % 2 == 0){
        factor = 1.0;
    }else{
        factor = -1.0;
    }

    // Non corretti
    // Problema di concorrenza in cui 2 thread scrivono nella stessa variabile 
    // in questo caso la somma. Busy_waiting ma spreca risorse
    
    // non tiene conto della concorrenza
    // for(i = posiz_iniziale; i <= posiz_finale; i++, factor = -factor){
        
    //     sum += factor / (2 * i + 1);
    // }

    // Concorrenza gestita male
    // for(i = posiz_iniziale; i <= posiz_finale; i++, factor = -factor){ 
    //     while (flag != my_rank);
    //     sum += factor / (2 * i + 1);
    //     flag = (flag + 1) % thread_count;
    // }

    for(i = posiz_iniziale; i <= posiz_finale; i++, factor = -factor){
        my_sum += factor / (2 * i + 1);
    }

    while (flag != my_rank);
    sum += 4*my_sum;
    flag = (flag + 1) % thread_count;       

    //sum *= 4.0;
    return NULL;
}

// Per una computazione più efficiente e che tiene conto della concorrenza utilizzo mutex
// vd. Pthread/Mutex.c
void *piGreco_deadlock(void *rank){

    long my_rank = (long) rank;
    int i;
    int local_n = n / thread_count;
    int posiz_iniziale = my_rank * local_n;
    int posiz_finale = (my_rank + 1) * local_n - 1;
    
    double my_sum = 0.0;
    double factor;

    if(posiz_iniziale % 2 == 0){
        factor = 1.0;
    }else{
        factor = -1.0;
    }

    for(i = posiz_iniziale; i <= posiz_finale; i++, factor = -factor){
        my_sum += factor / (2 * i + 1);
    }

    pthread_mutex_lock(&mutex);
    sum += 4*my_sum;
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}


sem_t semaphore;
// Per una computazione più efficiente e che tiene conto della concorrenza utilizzo semafori
void *piGreco_semaphore(void *rank){

    long my_rank = (long) rank;
    int i;
    int local_n = n / thread_count;
    int posiz_iniziale = my_rank * local_n;
    int posiz_finale = (my_rank + 1) * local_n - 1;
    
    double my_sum = 0.0;
    double factor;

    if(posiz_iniziale % 2 == 0){
        factor = 1.0;
    }else{
        factor = -1.0;
    }

    for(i = posiz_iniziale; i <= posiz_finale; i++, factor = -factor){
        my_sum += factor / (2 * i + 1);
    }

    sem_wait(&semaphore);
    sum += 4*my_sum;
    sem_post(&semaphore);
    
    return NULL;
}


int main(int argc, char const *argv[]){
    /* pi = 4(1 - 1/3 + 1/5 - 1/7 + ... + (-1)^n/(2n+1) + ...)  */
    long thread;
    pthread_t* thread_handles;

    // thread_count deve dividere n senza resto 
    thread_count = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);

    thread_handles = malloc(thread_count*sizeof(pthread_t));

    
    // Commenta le righe seguenti e dentro al for per utilizzare il semaforo o il mutex
    pthread_mutex_init(&mutex, NULL);
    sem_init(&semaphore, 0, 1);

    // Per ogni thread, partendo dal thread 0
    for (thread = 0; thread < thread_count; thread++){
        pthread_create(&thread_handles[thread], NULL, piGreco_deadlock, (void*) thread);
        pthread_create(&thread_handles[thread], NULL, piGreco_semaphore, (void*) thread);
    }
    
    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }
    // Rimozione del mutex
    pthread_mutex_destroy(&mutex);

    // Rimozione del semaforo
    sem_destroy(&semaphore);

    printf("Pi: %f\n", sum);
    free(thread_handles);

    return 0;
}
