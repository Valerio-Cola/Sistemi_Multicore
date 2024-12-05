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

### 📁 Cache
- **caching.md**: Documentazione sulla cache
- **ottimizzazione_perf.c**: Ottimizzazione delle performance
- **timing.c**: Misurazione dei tempi di esecuzione
- **my_timer.h**: Header per la gestione dei timer
- **perf_stat.md**: Statistiche delle performance

### 📁 CUDA
- **cuda_intro.cu**: Introduzione a CUDA
- **vect_x_vect.cu**: Moltiplicazione vettoriale con CUDA
- **cuda_intro.md**: Documentazione introduttiva su CUDA

## Compilazione

### MPI
```sh
mpicc file.c -o output
```

### Pthreads
```sh
gcc -pthread file.c -o output
```

### OpenMP
```sh
gcc -fopenmp file.c -o output
```

### CUDA
```sh
nvcc file.cu -o output
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

### OpenMP
```sh
./output
```

### CUDA
```sh
./output
```

## Tools

- **GDB** per debugging
- **Valgrind** per memory leak detection
- **Perf** per profiling
- **VS Code** con Markdown Preview Enhanced per documentazione
