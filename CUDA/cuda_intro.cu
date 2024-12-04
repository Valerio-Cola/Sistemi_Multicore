/*
CPU
Architettura specifica per eseguire codice in sequenziale per mettrci meno tempo possibile
Suddiviso in:
    cache: Convert long latency memory accesses to short latency cache accesses
    dram
    unità di controllo 
    unità di calcolo ALU

GPU
Architettura specifica per eseguire codice in parallelo, inizialmente incentrato sul rendering di immagini
Suddiviso in:
    90% del chip dedicato a unità di calcolo: meno potente dela cpu frequenza piu alta ma è possibile inserirne di piu
    Unita di controllo divise in più core: No branch prediction, In-order execution
    Small caches: To boost memory throughput
    DRAM: banda più alta più thread possono accedere alla memoria in parallelo

CPUs for sequential parts 
where latency matters
GPUs for parallel parts 
where throughput matters

Most used connection between CPU and GPU:        RAM                         
PCIe bus:                                         |
        1 linea per CPU, GPU, RAM   CPU - PCIe - GPU - DRAM

Un primo problema riguarda la gestione della memoria poichè la GPU e l'host non condividono la stessa
ed è quindi necessario un costante trasferimento
Inoltre le GPU potrebbero non seguire la stessa rappresentazione dei numeri a virgola mobile e avere la stessa accuratezza delle CPU

CUDA-core
Più core raggruppati in streaming multiprocessor SM (NVIDIA), condivide la stessa cache e la stessa unita di controllo,
2+ formano building block (NVIDIA) 
Memoria globale condivisa tra tutti SM
è quindi un architettura che permette l'esecuzione di molti thread in parallelo
Thread organizzati in una struttura 6D:
    Insieme di cubi detta griglia
    Ogni vertice ha un blocco con tot thread
    è possibile ottenere le coordinate di ogni thread
    Ogni GPU ha una capacità computazionale che indica le dimensioni massime di blocchi e griglia
    Le dimensioni si possono assegnare mediante dim3 vettore di interi:
        dim3 block(3,2); rettangolo 3x2
        dim grid(4,3,2); parallelepipedo 4x3x2
        foo<<<grid, block>>>();
        Quindi una griglia con 24 blocchi ognuno con 6 thread

Struttura di un programma cuda
    - Allocazione memoria GPU
    - Trasferimento dati da memoria host -> GPU
    - Esecuzione del kernel CUDA
    - Trasferimento risultati da memoria GPU -> host

Thread Scheduling
 Ogni thread viene eseguito su un processore di streaming (CUDA core). 
 I core sullo stesso SM condividono l'unità di controllo, cioè devono eseguire sincronicamente la stessa istruzione. 
 SM diversi possono eseguire kernel differenti. 
 Ogni blocco viene eseguito su un SM, quindi non posso avere un blocco che si estende su più SM, ma posso avere più blocchi che vengono eseguiti sullo stesso SM. 
 Una volta che un blocco è completamente eseguito, l'SM eseguirà il successivo. 
 Non tutti i thread in un gruppo vengono eseguiti contemporaneamente.



Warps
 Vengono eseguiti in gruppi chiamati warps (nelle GPU attuali, la dimensione di un warp è 32 – potrebbe cambiare in futuro, controllare la variabile warpSize)
 I thread in un blocco sono divisi in warps secondo il loro ID intra-blocco (cioè, i primi 32 thread in un blocco appartengono allo stesso warp, i successivi 32 thread a un warp diverso, ecc...)
 Tutti i thread in un warp vengono eseguiti secondo il modello Single Instruction, Multiple Data (SIMD)—cioè, in ogni istante, un'istruzione viene prelevata ed eseguita per tutti i thread nel warp. Di conseguenza, tutti i thread in un warp avranno sempre lo stesso timing di esecuzione.
 Diversi scheduler di warp (ad esempio, 4) possono essere presenti su ciascun SM. Cioè, più (ad esempio, 4) warps possono essere eseguiti allo stesso tempo, ciascuno possibilmente seguendo un percorso di esecuzione diverso


Divergenza dei Warps
 Tutti i thread in un warp vengono eseguiti secondo il modello Single Instruction, Multiple Data (SIMD), cioè in ogni istante, un'istruzione viene prelevata ed eseguita per tutti i thread nel warp.
 Di conseguenza, tutti i thread in un warp avranno sempre lo stesso timing di esecuzione.
 Cosa succede se il risultato di un'operazione condizionale li porta su percorsi diversi?
 Tutti i percorsi divergenti sono valutati (se i thread si diramano in essi) in sequenza finché i percorsi non convergono di nuovo.
 I thread che non seguono il percorso attualmente in esecuzione sono sospesi.

*/


#include <stdio.h>
#include <cuda.h>
/*
Compilazione:    nvcc -arch=sm_20 -o sample sample.cu
Esecuzione:      ./sample

Funzione che puo essere chiamata sia dall'host che GPU, ma deve necessariamente essere eseguita su GPU
No valori di ritorno, di base neache I/O
 __global__ indica che la funzione è eseguita su GPU
 __device__ indica che la funzione è eseguita su GPU ma può essere chiamata solo da altre funzioni eseguite su GPU
 __host__ indica che la funzione è eseguita su CPU */
_global__ void hello(){
    printf("Hello World from GPU!\n");
}


int main(){
    // Chiamata alla funzione hello in un singolo blocco con 10 thread, la chiamata è asincrona quindi bisogna sincronizzare la GPU = attendere che finisca l'esecuzione
    hello<<<1,10>>>();
    cudaDeviceSynchronize();
    
    //Bisogna combinare i valori di threadIdx e blockIdx per ottenere un id univoco per ogni thread, poichè due thread su blocchi diversi potrebbero avere stesso threadIdx
    int myID = (blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) 
                * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
    return 0;
}
