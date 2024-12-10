# Shared memory

## Stencil 1D:
    Dato array 1D ogni elemento è calcolato mediante se stesso e i tot elementi alla sua destra e sinistra.

    Un thread per elemento 

    Ogni elemento viene letto più volte per il calcolo dei vicini, si avranno più thread che cercano di leggere lo stesso elemento dalla memoria globale, è quindi più comodo spostarlo nella shared memory.

Constant memory
    Cached
    Supporta il broadcasting a tutti i thread nel warp

    ponendo di avere piu warp i quali devono accedere alla stessa variabile:

    1. Se su mem globale: il primo la invia alla L2, gli altri warp potrebbero non trovarla in L2 e andarala a cercare in globale, questo perchè potrebbero accedervi troppo in ritardo e intanto il primo warp ha cacheato altri valori

    2. Se su memoria costante: il primo warp la copia nella mem cache-costante, poichè vengono scritti dati con poca frequenza è più probabile che venga trovata dagli altri warp

    __constant_ type variable_name; // static
    cudaMemcpyToSymbol(variable_name, &host_src, sizeof(type), cudaMemcpyHostToDevice);
    warning: cannot be dynamically allocated
    data will reside in the constant memory address space
    • has static storage duration (persists until the application ends)
    • readable from all threads of a kernel

Conversione immagine in bw
Per ogni pixel allocato in una matrice 2D: red*0.21 + green*0.72 + blue*0.07

la matrice può essere vista come un array composto dalle righe della matrice messe in sequenza
accesso a singolo pixel [riga*len_riga+colonna]


Prestazioni GPU:
Calcolate mediante il n
umero operazioni in virgola mobile al secondo FLOP/s, il valore va scalato in base al numero di bit

