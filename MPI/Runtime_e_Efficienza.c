#include <mpi.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{

    // Inizializzazione di MPI
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /*
    Runtime di ogni processo

    double inizio, fine;    
    start = MPI_Wtime();

        ...code...

    finish = MPI_Wtime();
    printf("Processo %d concluso in %e secondi \n", rank, fine-inizio);

    */

    // Massimo runtime tra i vari processi
    double inizio, fine, runtime, runtime_max;
    inizio = MPI_Wtime();

        /*    code    */   

    fine = MPI_Wtime();
    runtime = fine - inizio;

    MPI_Reduce(&runtime, &runtime_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Il processo più lento è di %e secondi\n", runtime_max);
    }
    
    /*
        Tempo di computazione
        
        Il codice considera anche il tempo di attesa dei processi, il tempo finale non sarà quello di effettiva computazione (Noise)

        Per ovviare a ciò devono partire tutti insieme, utilizzo MPI_Barrier:
            Permette di "sincronizzare" i processi e farli partire tutti in una finestra temporale minima 
        
        Altro problema se eseguo più volte lo stesso codice avrò runtime differenti ogni volta poichè:
            -Interferenze causate dal sistema che è condiviso quindi context switch, cache pollution,...
            -Interferenze nella rete

        Quindi è più corretto riportare la distribuzione dei runtime massimi per ogni esecuzione.

        Si nota come all'aumentare della dimensione del problema aumenta il tempo di computazione, 
        maggiori sono i processi e minore è il runtime, ma non sempre è cosi:

        Definiamo:
            T_seriale(n)      -> Tempo esecuzione seriale per un problema di lunghezza n
            T_parallelo(n,p)  -> Tempo esecuzione parallela per un problema di lunghezza n con p processi    
            S(n,p)            -> Speedup = T_seriale(n)/T_parallelo(n,p)
                Se S(n,p) = p -> Speedup lineare e ideale, misura efficienza del calcolo parallelo rispetto al calcolo sequenziale 
                
            E(n,p)            -> Efficienza = S(n,p)/p  
                Idealmente vogliamo =1 ma più tende a zero peggio è. Più è piccolo il problema più tende a zero  
        
        Effettuo lo scaling:
            Strong: Aumenta processi diminuisce dimensione problema. 
            Weak: Aumenta dimensione problema e numero processi.

        Amdahl’s Law
            Ci sono sempre parti di programma che non possono essere parallelizzate
            Lo speedup è limitato dalla serial fraction α (frazione temporale)    
    
            Quindi abbiamo: T_parallelo(p) = (1-α)T_seriale + α*T_seriale / p
            La parte 1-α deve necessariamente essere sequenziale
            Da ciò ricavo: S(p) = T_seriale / (1-α)T_seriale + (α*T_seriale / p)
            Con limite p che tende a infinito = 1 / (1-α)
                Notiamo quindi se il 60% dell'applicazione può essere parallelizzata, α=0.6 e ci si può aspettare uno speedup di almeno 2.5
    
        Gustafson’s Law
           Scaled speedup = Considerando il weak scaling, α aumenta con la dimensione del problema 
            S(n,p) = (1-α) + αp    
    */      
    
    MPI_Finalize();
    return 0;
}
