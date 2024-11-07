# Repository Programmazione Parallela
Questa repository è una raccolta di appunti universitari del corso **Programmazione di Sistemi Multicore**. Contiene esempi di codice, esercizi e spiegazioni teoriche riguardanti la programmazione parallela e concorrente utilizzando MPI (Message Passing Interface) e Pthreads (POSIX Threads).

## Struttura Repository

### 📁 MPI (Message Passing Interface)
- **introduzione.c**: Concetti base di MPI, inizializzazione e comunicazione
- **Bloccanti_e_Non.c**: Comunicazione bloccante e non-bloccante tra processi
- **Datatype.c**: Gestione dei tipi di dato derivati in MPI
- **Input_e_Reduce.c**: Input distribuito e operazioni di riduzione
- **Scatter_Gather.c**: Distribuzione e raccolta dati tra processi
- **AllToAll.c**: Comunicazione tutti-a-tutti
- **Sorting.c**: Algoritmi di ordinamento paralleli
- **Runtime_e_Efficienza.c**: Analisi delle performance e scaling
- **Gather_Bcast_Matrici.c**: Operazioni su matrici distribuite

### 📁 Pthread (POSIX Threads)
- **introduzione.c**: Concetti base dei thread POSIX
- **Mutex.c**: Sincronizzazione con mutex
- **Semafori.c**: Sincronizzazione con semafori
- **rwlock_lista_puntata.c**: Liste concatenate thread-safe con read-write lock
- **thread_safety.c**: Gestione della thread safety
- **thread_pinning.c**: Assegnamento thread a core specifici
- **piGreco_concorrenza.c**: Calcolo parallelo di π con diverse strategie
- **matrice_vettore.c**: Moltiplicazione matrice-vettore parallela

### 📁 Cache
- **my_timer.h**: Utility per misurare i tempi di esecuzione
- **timing.c**: Esempi di misurazione performance
- **ottimizzazione_perf.c**: Ottimizzazione accessi in memoria
- **caching.md**: Documentazione su gerarchia della cache
- **perf_stat.md**: Guida all'analisi delle performance

## Compilazione

### MPI
```sh
mpicc file.c -o output
```

### Pthreads
```sh
gcc -pthread file.c -o output
```

## Esecuzione

### MPI
```sh
mpirun -n <num_processi> ./output
```

### Pthreads
```sh
./output <num_threads>
```

## Tools

- **GDB** per debugging
- **Valgrind** per memory leak detection
- **Perf** per profiling
- **VS Code** con Markdown Preview Enhanced per documentazione
