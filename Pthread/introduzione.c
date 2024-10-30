#include <stdio.h>
#include <pthread.h>

// POSIX Threads (Solo su posix systems)
// A process is an instance of a running (or suspended) program.
// Threads are analogous to a “light-weight” process.
// In a shared memory program a single process may 
// have multiple threads of control.
// Differentemente da MPI i thread vengono avviati dal programma

// Compilazione: gcc −g −Wall −o pth_hello pth_hello . c −lpthread
// Run: pth_hello   <number of threads>
void my_function(int x)
{
    printf("Input funzione: %d\n", x);
}

// Utilizzare void* come argomentogli permette can point to a list containing one or more 
// values needed by thread_function
void my_function2(void* x)
{
    int* x_ptr = (int*) x;
    printf("Value of x: %d\n", x_ptr);
}

int main(int argc, char const *argv[])
{
    // Funzione di avvio thread
    // Opaque
    // The actual data that they store is system-specific.
    // Their data members aren’t directly accessible to user code. 
    // However, the Pthreads standard guarantees that a pthread_t object does store 
    // enough information to uniquely identify the thread with which it’s associated

    int pthread_create( pthread_t *thread,               // Puntatore alla struttura che identifica il thread
                        const pthread_attr_t *attr,      // NULL, attributi del thread
                        void *(*start_routine) (void *), // Puntatore alla funzione che il thread eseguirà
                        void *arg);                      // Argomento della funzione
                                                         // In questo caso la funzione ritorna void* e come argomento ha void*

    // Creazione di un puntatore a funzione vd. riga 12
    void(*func_ptr)(int) = my_function;
    *func_ptr(10);
    printf("Puntatore: %p\n", func_ptr);


    // Creazione di un thread con funzione e puntatore a variabile come argomento vd. riga 19
    int x = 5;
    pthread_create(NULL,NULL,my_function2,(void*) &x);

    // Creazione di un thread con funzione e valore intero come argomento
    pthread_create(NULL,NULL,my_function,10);

    // La chiamata a questa funzione fa attendere la terminazione del thread passato come argomento
    int pthread_join(pthread_t thread, void **value_ptr);

    // Il modo corretto per fare la join di più thread
    int num_thread = 10;
    for(int i = 0; i < num_thread; i++){
        pthread_create(NULL,NULL,my_function,(void*) &i);
    }

    for(int i = 0; i < num_thread; i++){
        pthread_join(NULL,NULL);
    }

    // Provides the thread ID of the calling thread
    pthread_t pthread_self(void);

    // compares thread ID
    int pthread_equal(pthread_t t1, pthread_t t2);

    // Identificazione thread
    pthread_t self = pthread_self();
    printf("Thread ID: %lu\n", self);

    return 0;
}
 