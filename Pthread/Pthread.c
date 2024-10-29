#include <stdio.h>
#include <pthread.h>

void my_function(int x)
{
    printf("Value of x: %d\n", x);
}

void my_function2(void *x)
{
    int *y = (int*) x;
    printf("Value of x: %d\n", *y);
}

int main(int argc, char const *argv[])
{
    int pthread_create( pthread_t *thread,               // Puntatore alla struttura che identifica il thread
                        const pthread_attr_t *attr,      // NULL, attributi del thread
                        void *(*start_routine) (void *), // Puntatore alla funzione che il thread eseguirà
                        void *arg);                      // Argomento della funzione


    // Creazione di un thread con funzione e puntatore a variabile come argomento  
    int x = 5;
    pthread_create(NULL,NULL,my_function2,(void*) &x);

    // Creazione di un thread con funzione e valore intero come argomento
    pthread_create(NULL,NULL,my_function,10);

    // Creazione di un puntatore a funzione
    void(*func_ptr)(int) = my_function;
    func_ptr(10);
    printf("Address of function: %p\n", func_ptr);


    int num_thread = 10;
    for(int i = 0; i < num_thread; i++){
        pthread_create(NULL,NULL,my_function,(void*) &i);
    }

    for(int i = 0; i < num_thread; i++){
        // Funzione che blocca il thread principale fino a quando il thread passato come argomento non termina
        pthread_join(NULL,NULL);
    }
    
    // Identificazione thread
    pthread_t self = pthread_self();
    printf("Thread ID: %lu\n", self);

    return 0;
}
 