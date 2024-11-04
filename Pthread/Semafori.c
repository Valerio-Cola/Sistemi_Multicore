#include <stdio.h>  
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

#define MSG_MAX 100

int thread_count;
char** messages;
//completa il codice allocando e inizializzando messages con tutti null e semaphores con 0 (locked) nel main. Completa le ulteriori parti mancanti
// Scambio di msg tra thread
// Funzione eseguita dai thread per inviare messaggi
void* Send_msg(void* rank){
    long my_rank = (long) rank; // Ottiene il rank del thread corrente
    long dest = (my_rank + 1) % thread_count; // Calcola il destinatario del messaggio
    char* my_msg = malloc(MSG_MAX * sizeof(char)); // Alloca memoria per il messaggio

    // Crea il messaggio
    sprintf(my_msg, "Ciao a %ld da %ld", dest, my_rank);
    messages[dest] = my_msg; // Invia il messaggio al destinatario
    sem_post(&semaphores[dest]);

    // Controlla se il thread ha ricevuto un messaggio
    sem_wait(&semaphores[my_rank]);
    printf("Thread %ld > %s\n", my_rank, messages[my_rank]); // Stampa il messaggio ricevuto
    return NULL; // Termina la funzione
}

// Inizializza semaforo con valore iniziale, è un unsigned int
// Blocco -> Sblocco dallo stesso thread
// Incremento/Decremento da diversi thread
int sem_init(sem_t *sem, int pshared, unsigned int value);

// Decrementa il semaforo, se il valore è 0 il thread viene bloccato
int sem_wait(sem_t *sem);

// Incrementa il semaforo e se c'è un thread in wait lo sblocca e procede
int sem_post(sem_t *sem);

// Ottiene il valore del semaforo
int sem_getvalue(sem_t *sem, int *sval);

// Rimuove il semaforo
int sem_destroy(sem_t *sem);

int main(int argc, char const *argv[])
{
    // Busy-waiting enforces the order threads access a 
    // critical section.

    // Using mutexes, the order is left to chance and 
    // the system.

    // There are applications where we need to control 
    // the order threads access the critical section
    
    return 0;
}
