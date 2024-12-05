/*
OpenMP mira a decomporre un programma sequenziale in 
componenti che possono essere eseguiti in parallelo

OpenMP consente una conversione "incrementale" dei 
programmi sequenziali in paralleli, con l'assistenza del compilatore

Pragma
• Istruzioni speciali del preprocessore.
• Tipicamente aggiunte a un sistema per consentire comportamenti che 
non fanno parte della specifica base del C.
• I compilatori che non supportano i pragma li ignorano.
#pragma


Gestione del numero di thread, in ordine di precedenza (omp_set_num_threads sovrascrive OMP_NUM_THREADS):

Modifica della variabile d’ambiente OMP_NUM_THREADS:
$ echo ${OMP_NUM_THREADS}  # to query the value
$ export OMP_NUM_THREADS=4 # to set it in BASH

Imposta il numero di thread da utilizzare 
omp_set_num_threads(int num_threads)

num_threads: numero di thread da utilizzare


Mutual exclusion

lock & unlock
# pragma omp critical

Protetta solo l'assegnazione della variabile
# pragma omp atomic



In OpenMP, the scope of a variable refers to 
the set of threads that can access the variable 
in a parallel block
shared: all threads can access the variable
    declared outside pragma omp parallel
private: each thread has its own copy of the variable
    declared inside pragma omp parallel
reduction: specifica una variabile condivisa e un operatore di riduzione
    la variabile condivisa è inizializzata al valore identità dell'operatore
    l'operatore è applicato tra la variabile condivisa e la variabile privata 




Una riduzione è un calcolo che applica ripetutamente lo 
stesso operatore di riduzione a una sequenza di operandi per ottenere un singolo risultato. 
Tutti i risultati intermedi dell'operazione devono essere memorizzati nella stessa variabile:
la variabile di riduzione. 

reduction(operator:variable)  (operators: +, *, -, &, |, ^, &&, ||)

Crea una variabile privata inizializzata al valore identità dell'operatore 
Con una sezione critica o atomica, aggiorna la variabile con il risultato dell'operazione tra la variabile privata e la variabile condivisa



Clausole di modifica dello scope
Permette al programmatore di specificare lo scope di ogni variabile in un blocco pragma.
default()
    shared: tutte le variabili sono condivise
    private: ogni thread ha la propria copia delle variabili
    none: tutte le variabili devono avere scope esplicito
    reduction:op:var: tutte le variabili sono private, eccetto var
        che è una variabile di riduzione con operatore op
    firstprivate: tutte le variabili sono private, eccetto var
        che è privata e inizializzata con il valore della
        variabile fuori dal blocco 
    lastprivate: tutte le variabili sono private, eccetto var che è
        privata e il cui valore viene copiato nella variabile
        fuori dal blocco
    threadprivate: tutte le variabili sono private, eccetto var che è privata
        e il cui valore viene copiato nella variabile fuori dal blocco
    copyin: tutte le variabili sono private, eccetto var che è privata e
        inizializzata con il valore della variabile fuori dal blocco
    copyprivate: tutte le variabili sono private, eccetto var che è privata e
        inizializzata con il valore della variabile fuori dal blocco 
*/

nt x = 5; 
// Ogni thread ha la propria copia di x che non è quella inizializzata fuori dal blocco pragma
#pragma omp parallel private(x) { 
    x = x+1; // dangerous (x not initialized) 
    printf("thread %d: private x is %d\n",omp_get_thread_num(),x);
}
// Fuori dal blocco pragma x è quella dichiarata all'inizio 5
printf("after: x is %d\n",x)


//  gcc −g −Wall −fopenmp −o omp_ omp.c
//  ./omp_hello 4

// Se la variabile viene definita dal compilatore allora omp è supportato   
// Allora importalo
#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>


void Hello(void) {

    // Se omp è supportato allora esegui il blocco di codice altrimenti esegui il blocco else
    # ifdef _OPENMP
       // Si ottiene il rank (id) del thread chiamante e il numero di thread attivi 
       int my_rank = omp_get_thread_num ( );
       int thread_count = omp_get_num_threads ( );
    # else
       int my_rank = 0;
       int thread_count = 1;
    # endif

    printf("Hello from thread %d of %d\n", my_rank, thread_count);
}

int main(int argc, char const *argv[])
{   
    int thread_count = strtol(argv[1], NULL, 10);

    // Esegue funzione Hello() in parallelo con thread_count thread
    #pragma omp parallel num_threads(thread_count)
    Hello();
    
    return 0;
}
