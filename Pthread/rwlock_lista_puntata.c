#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>


// Definizione di una lista concatenata
void linked_lists() {
    struct list_node {
        int data;
        struct list_node* next;
    };
} 

//controllare se un elemento è presente nella lista
int Member(int value, struct list_node* head_p) {
    // head_p = puntatore alla testa della lista
    // value = valore da cercare
    // curr_p = puntatore all'elemento corrente
    struct list_node_s* curr_p = head_p;
    
    // scorre la lista finché non trova l'elemento o arriva alla fine
    while(curr_p != NULL && curr_p->data < value) {
        curr_p = curr_p->next;
    }

    // se l'elemento non è presente o è maggiore di value
    if(curr_p == NULL || curr_p->data > value) {
        // l'elemento non è presente
        return 0;
    } else {
        // l'elemento è presente
        return 1;
    }
}


//inserire un elemento nella lista
int Insert(int value, struct list_node_s** head_pp) {
    // head_pp = puntatore al puntatore alla testa della lista
    // curr_p = puntatore all'elemento corrente
    // pred_p = puntatore all'elemento precedente
    struct list_node_s* curr_p = *head_pp;
    struct list_node_s* pred_p = NULL;
    struct list_node_s* temp_p;

    // scorre la lista finché non trova l'elemento o arriva alla fine
    while(curr_p != NULL && curr_p->data < value) {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    // se l'elemento è già presente
    if(curr_p == NULL && curr_p->data == value) {
        // alloca memoria per il nuovo elemento
        temp_p = malloc(sizeof(struct list_node_s)); 
        
        // inizializza il nuovo elemento
        temp_p->data = value; 
        
        // il nuovo elemento punta all'elemento corrente
        temp_p->next = curr_p;

        if (pred_p == NULL) { 
            // il puntatore alla testa punta al nuovo elemento
            *head_pp = temp_p;
        } else {
            // il precedente punta al nuovo elemento e viene inserito in mezzo
            pred_p->next = temp_p;
        }
        return 1;
    } else { 
        // elemento già nella lista
        return 0;
    }
}

// Rimozione un elemento dalla lista
int Delete(int value, struct list_node_s** head_pp) {
    struct list_node_s* curr_p = *head_pp;
    struct list_node_s* pred_p = NULL;

    // scorre la lista finché non trova l'elemento o arriva alla fine
    while(curr_p != NULL && curr_p->data < value) {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if (curr_p != NULL && curr_p->data == value) {
        // elemento in testa è nullo
        if (pred_p == NULL) { 
            // il puntatore alla testa punta al successivo
            *head_pp = curr_p->next;
            free(curr_p);
        // elemento in mezzo
        } else { 
            // il precedente punta al successivo
            pred_p->next = curr_p->next;
            free(curr_p);
        }
        return 1;
    } else { 
        // elemento non presente
        return 0;
    }
}

/*
Immaginiamo di avere più thread che accedono alla stessa lista. per condividere l'accesso a tale lista, rendiamo la variabile head_p globale.
Quando più thread chiamano Member nello stesso momento tutto è a posto
Cosa succede se un thread chiama Member mentre un altro thread sta eliminando un elemento?
*/

// Soluzione 1: usare un mutex per proteggere l'accesso alla lista
//Pthread_mutex_lock(&list_mutex);
//Member(value, head_p);
//Pthread_mutex_unlock(&list_mutex);
//in questo modo, solo un thread alla volta può accedere alla lista (serializzazione dell'accesso alla lista)
//quando la maggior parte delle operazioni sono Member, non stiamo sfruttando il parallelismo
//se invece la maggior parte delle operazioni sono Insert/Delete, il parallelismo è ridotto, ma è la soluzione migliore

// Soluzione 2: invece di bloccare l'intera lista, possiamo bloccare solo i nodi interessati
// (finer grade approach)
struct list_node_s {
    int data;
    struct list_node_s* next;
    pthread_mutex_t mutex;
};
// è molto più complessa e molto più lenta, perché ogni volta che si accede a un nodo, bisogna bloccare e sbloccare una mutex
// viene anche incrementato lo spazio in memoria necessario per la lista

// Soluzione 3: usare un read-write lock
/*
un read-wrte lock è simile ad una mutex ma fornisce due funzioni di lock
- una per la lettura (read lock)
- una per la scrittura (write lock)
quando un thread ha un read lock, altri thread possono avere un read lock
quando un thread ha un write lock, nessun altro thread può avere un read o write lock
*/
int main(int argc, char const *argv[]){
    
    pthread_rwlock_t rwlock;
    
    // inizializza il read-write lock
    int pthread_rwlock_init(pthread_rwlock_t* rwlock, const pthread_rwlockattr_t* attr);
    pthread_rwlock_init(&rwlock, NULL);
    
    // blocca il read-write lock in modalità lettura
    pthread_rwlock_rdlock(&rwlock);
    
    // sblocca il read-write lock
    pthread_rwlock_unlock(&rwlock);
    
    // blocca il read-write lock in modalità scrittura
    pthread_rwlock_wrlock(&rwlock);
    
    // sblocca il read-write lock
    pthread_rwlock_unlock(&rwlock);
    
    // distrugge il read-write lock
    pthread_rwlock_destroy(&rwlock);

    return 0;

}