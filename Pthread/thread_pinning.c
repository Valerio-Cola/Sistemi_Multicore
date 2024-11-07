#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

/*

 * Questa funzione configura l'affinità di un thread in modo che venga eseguito
 * specificatamente sul core CPU 3.

 * 1. Creazione di una struttura CPU set
 * 2. Recupero dell'ID del thread corrente
 * 3. Azzeramento del CPU set
 * 4. Aggiunta del core CPU 3 al set
 * 5. Impostazione dell'affinità del thread al core specificato

*/
void* thread_func(void* thread_args){
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();
    
    /* Set affinity mask to include core 3 */
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);

    int s = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}
