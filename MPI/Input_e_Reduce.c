#include <mpi.h> 
#include <stdio.h>  

// Funzione per ottenere input dai processi
void Get_input(int rank, int size, double *a, double *b, int *n) {
    
    // Se il processo è il processo root (rank 0), ottiene l'input dall'utente
    if(rank == 0){
        printf("Enter a, b and n \n");
        scanf("%lf %lf %d", a, b, n);

        // Invia i valori di a, b, e n agli altri processi, size è il numero di processi
        for(int dest = 1; dest < size; dest++){
            MPI_Send(a, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(b, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        // Se non è il processo root, riceve i valori di a, b, e n dal processo root
        MPI_Recv(a, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Funzione alternativa per ottenere input utilizzando MPI_Bcast, è più efficiente
void Get_input2(int rank, int size, double *a, double *b, int *n){
    if(rank == 0){
        // Se il processo è il processo root (rank 0), ottiene l'input dall'utente
        printf("Inserisci a, b e n\n");
        scanf("%lf %lf %d", a, b, n);
    }
    // Broadcast dei valori di a, b, e n a tutti i processi
    MPI_Bcast(a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

// Funzione per calcolare l'integrale utilizzando la regola del trapezio
double Trap(double left_endp, double right_endp, int trap_count, double base_len){
    double estimate, x;  // 'estimate' accumula il valore dell'integrale, 'x' è il punto di valutazione corrente

    // Calcolo del primo e ultimo valore dell'intervallo
    estimate = (f(left_endp) + f(right_endp)) / 2.0;
    // Somma dei valori intermedi
    for(int i = 1; i <= trap_count - 1; i++){
        x = left_endp + i * base_len;
        estimate += f(x);
    }
    // Moltiplicazione della somma per la lunghezza della base
    estimate *= base_len;
    return estimate;
}

int main(int argc, char const *argv[])
{
    // Variabili per il numero di processi e il rank del processo corrente
    int size, rank;

    // Inizializzazione dell'ambiente MPI
    MPI_Init(NULL, NULL);

    // Ottiene il rank del processo corrente
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Ottiene il numero totale di processi
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Variabili per l'input dell'utente
    int n;  // Numero di trapezi per l'integrazione
    double a, b;  // Estremi dell'intervallo di integrazione
    double h;  // Lunghezza della base dei trapezi

    // Variabili per i calcoli locali (singoli processi)
    int n_locale;  // Numero di trapezi per processo
    double a_locale, b_locale;  // Estremi locali dell'intervallo di integrazione

    // Il singolo processo ottiene l'input utilizzando la funzione Get_input2
    Get_input2(rank, size, &a, &b, &n);

    // Calcolo della lunghezza della base, la base del trapezio è uguale per tutti
    h = (b - a) / n;
    // Numero di trapezi per processo
    n_locale = n / size;

    // Calcola gli intervalli locali per ogni processo
    a_locale = a + rank * n_locale * h;
    b_locale = a_locale + n_locale * h;

    // Calcola l'integrale locale
    double int_locale = Trap(a_locale, b_locale, n_locale, h);

    // Riduce tutti i risultati locali in un unico risultato totale sommando in parallelo tutti i Reduce chiamati dai processi
    /*
        MPI_MAX     Massimo
        MPI_MIN     Minimo
        MPI_SUM     Somma
        MPI_PROD    Prodotto
        MPI_LAND    AND logico
        MPI_LOR     OR logico
        MPI_LXOR    XOR logico
        MPI_BAND    Bitwise AND
        MPI_BOR     Bitwise OR
        MPI_BXOR    Bitwise XOR
    */
    double int_totale;
    // Ogni processo prende l'indirizzo della variabile calcolata e la somma a int_totale
    MPI_Reduce(&int_locale, &int_totale, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Se è il processo root, stampa il risultato
    if(rank == 0){
        printf("Con n = %d trapezoidi\n", n);
        printf("Integrale da %f a %f = %.15e \n", a, b, int_totale);
    }

    // Finalizza l'ambiente MPI
    MPI_Finalize();
    return 0;
}
