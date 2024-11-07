#include <sys/time.h>
#include <stddef.h>
#include <stdio.h>

/*  Macro per ottenere il timestamp corrente in secondi 
    Utilizza la struct timeval e la funzione gettimeofday 
    now: variabile dove verrà salvato il timestamp 
   
    Il \ alla fine di ogni riga permette di scrivere una macro su più righe
    Senza i \, il preprocessore C interpreterebbe ogni nuova
    riga come la fine della macro
*/


#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    /* Converte in secondi */ \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

int main(){
    double start, finish, elapsed;
    GET_TIME(start);
    
    // Code to be timed
    
    GET_TIME(finish);
    elapsed = finish - start;
    
    printf("The code to be timed took %e micro seconds\n", elapsed);
}