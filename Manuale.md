# MPI

## Inizializzazione MPI con comunicatore
```c
int r = MPI_Init(NULL, NULL);
int size, rank;
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```

## Chiusura MPI
```c
MPI_Finalize();
```

## Lancio errore
```c
MPI_Abort(MPI_COMM_WORLD, r);
```

## Bloccanti
```c
int MPI_Send(
	void* msg_buf_p,          // Puntatore ai dati
	int msg_size,             // Numero di elementi nel messaggio
	MPI_Datatype msg_type,    // Tipo di dato nel messaggio
	int dest,                 // Rank processo a cui inviare
	int tag,
	MPI_Comm communicator     // Comunicatore
);

int MPI_Recv(
	void* msg_buf_p,          // Puntatore ai dati
	int buf_size,
	MPI_Datatype buf_type,
	int source,               // Rank processo da cui riceve
	int tag,
	MPI_Comm communicator,
	MPI_Status* status_p      // Utilizziamo MPI_STATUS_IGNORE
);
```

## Non bloccanti
- `Isend`
- `Irecv`

## Tag
- `MPI_ANY_SOURCE` è possibile ricevere da chiunque
- `MPI_ANY_TAG` è possibile ricevere msg con qualsiasi tag

## Tipi di dato
- `MPI_CHAR`
- `MPI_SIGNED_CHAR`
- `MPI_UNSIGNED_CHAR`
- `MPI_BYTE`
- `MPI_WCHAR`
- `MPI_SHORT`
- `MPI_UNSIGNED_SHORT`
- `MPI_INT`
- `MPI_UNSIGNED`
- `MPI_LONG`
- `MPI_UNSIGNED_LONG`
- `MPI_LONG_LONG_INT`
- `MPI_UNSIGNED_LONG_LONG`
- `MPI_FLOAT`
- `MPI_DOUBLE`
- `MPI_LONG_DOUBLE`

## Comunicazione tra processi
Ogni processo utilizza due buffer invio e ricezione per scambiarsi dati:
```c
MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);
```

## Inizializzo richiesta
```c
MPI_Request request_send_r, request_send_l;
```

Da inserire dopo le `send` e `recv` per bloccare il processo affinché non vengono completate:
```c
MPI_Wait(&request_send_l, MPI_STATUS_IGNORE);
MPI_Wait(&request_send_r, MPI_STATUS_IGNORE);
```

## Richieste inizializzate mediante lista
```c
MPI_Request lista_richieste[4];
```

Attende che tutte le `recv` e `send` della lista vengano completate:
```c
MPI_Waitall(4, lista_richieste, MPI_STATUS_IGNORE);
```

## Broadcast
Broadcast dei valori della variabile `a` a tutti i processi:
```c
MPI_Bcast(a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

## Reduce
Ogni processo prende l'indirizzo della variabile calcolata e la somma a `int_totale`:
```c
MPI_Reduce(&int_locale, &int_totale, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

### Operazioni di riduzione
- `MPI_MAX`     Massimo
- `MPI_MIN`     Minimo
- `MPI_SUM`     Somma
- `MPI_PROD`    Prodotto
- `MPI_LAND`    AND logico
- `MPI_LOR`     OR logico
- `MPI_LXOR`    XOR logico
- `MPI_BAND`    Bitwise AND
- `MPI_BOR`     Bitwise OR
- `MPI_BXOR`    Bitwise XOR

## Calcolo tempo esecuzione
Ritornano dei `double`:
```c
inizio = MPI_Wtime();
/*    code    */
fine = MPI_Wtime();
```

## Scatter
Root invia il vettore spezzato in parti uguali:
```c
MPI_Scatter(a, len_vect / size, MPI_INT, MPI_IN_PLACE, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
```

Ogni processo dovrà farlo per ottenere il vettore `a`:
```c
MPI_Scatter(NULL, len_vect / size, MPI_INT, a, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
```

## Gather
Ogni processo invia il proprio pezzo di vettore `c` al vettore finale:
```c
MPI_Gather(c, len_vect / size, MPI_INT, finale, len_vect / size, MPI_INT, 0, MPI_COMM_WORLD);
```

## Broadcast del vettore
Broadcast del vettore `x` a tutti i processi:
```c
MPI_Bcast(x, colonne, MPI_INT, 0, MPI_COMM_WORLD);
```

## Strutture
```c
MPI_Datatype t;
int block_lengths[3] = {1, 1, 1}; // Numero di elementi per tipo
MPI_Aint displacements[3] = {0, 16, 24}; // Offset di memoria
MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
```

Genera struttura:
```c
MPI_Type_create_struct(3, block_lengths, displacements, types, &t);
```

Necessario per essere ottimizzato per le comunicazioni:
```c
MPI_Type_commit(&t);
```

Libera memoria struttura:
```c
MPI_Type_free(&t);
```

## PTHREAD

### Creazione di un Thread
```c
int pthread_create(
	pthread_t *thread,                  // Puntatore alla struttura che identifica il thread
	const pthread_attr_t *attr,         // NULL o attributi del thread
	void *(*start_routine) (void *),    // Puntatore alla funzione che il thread eseguirà
	void *arg                           // Argomento della funzione
);
```
Esempio di utilizzo:
```c
pthread_create(NULL, NULL, my_function, (void*)10);
```
Questa chiamata crea un nuovo thread che esegue `my_function` con l'argomento `10`.

### Attesa della Terminazione di un Thread
```c
int pthread_join(pthread_t thread, void **value_ptr);
```
Questa funzione fa attendere il thread chiamante fino alla terminazione del thread specificato.

### Identificazione del Thread Corrente
```c
pthread_t pthread_self(void);
```
Restituisce l'ID del thread chiamante.

### Confronto di ID di Thread
```c
int pthread_equal(pthread_t t1, pthread_t t2);
```
Confronta gli ID di due thread e restituisce un valore diverso da zero se sono uguali.

### Semafori

#### Inizializzazione del Semaforo
```c
int sem_init(sem_t *sem, int pshared, unsigned int value);
```
Inizializza un semaforo con un valore iniziale.

#### Attesa su un Semaforo
```c
int sem_wait(sem_t *sem);
```
Decrementa il semaforo. Se il valore è 0, il thread viene bloccato fino a quando il semaforo non viene incrementato.

#### Segnalazione su un Semaforo
```c
int sem_post(sem_t *sem);
```
Incrementa il semaforo e, se ci sono thread in attesa, ne sblocca uno.

#### Ottenere il Valore del Semaforo
```c
int sem_getvalue(sem_t *sem, int *sval);
```
Recupera il valore corrente del semaforo.

#### Distruzione del Semaforo
```c
int sem_destroy(sem_t *sem);
```
Rimuove il semaforo e libera le risorse.

### RWLock (Read-Write Lock)

#### Inizializzazione del RWLock
```c
int pthread_rwlock_init(pthread_rwlock_t* rwlock, const pthread_rwlockattr_t* attr);
```
Inizializza un read-write lock.

#### Bloccare in Modalità Lettura
```c
int pthread_rwlock_rdlock(pthread_rwlock_t* rwlock);
```
Blocca il rwlock per letture condivise.

#### Bloccare in Modalità Scrittura
```c
int pthread_rwlock_wrlock(pthread_rwlock_t* rwlock);
```
Blocca il rwlock per scritture esclusive.

#### Sbloccare il RWLock
```c
int pthread_rwlock_unlock(pthread_rwlock_t* rwlock);
```
Sblocca il rwlock.

#### Distruzione del RWLock
```c
int pthread_rwlock_destroy(pthread_rwlock_t* rwlock);
```
Distrugge il rwlock e libera le risorse.

### Mutex

#### Inizializzazione del Mutex
```c
int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);
```
Inizializza un mutex.

#### Bloccare il Mutex
```c
int pthread_mutex_lock(pthread_mutex_t *mutex);
```
Blocca il mutex. Se il mutex è già bloccato, il thread chiamante viene messo in attesa.

#### Sbloccare il Mutex
```c
int pthread_mutex_unlock(pthread_mutex_t *mutex);
```
Sblocca il mutex.

#### Distruzione del Mutex
```c
int pthread_mutex_destroy(pthread_mutex_t *mutex);
```
Rimuove il mutex e libera le risorse.

#### Esempio di Utilizzo del Mutex
```c
pthread_mutex_lock(&mutex);
sum += 4 * my_sum;
pthread_mutex_unlock(&mutex);
```
Questo esempio dimostra come proteggere una sezione critica utilizzando un mutex.

## OpenMPI

### Esecuzione della funzione `Hello()` in parallelo con `num_thread` thread

```c
#pragma omp parallel num_threads(num_thread)
Hello();
```

Oppure, puoi impostare il numero di thread in questo modo:

```c
omp_set_num_threads(int num_threads);
```

### Clausole di Condivisione delle Variabili

Quando aggiungi una pragma, puoi specificare come le variabili sono condivise tra i thread:

- **private(variabile)**: Ogni thread ha la propria copia della variabile, indipendente da quella esterna.
- **shared**: Tutte le variabili sono condivise tra i thread.
- **none**: Nessuna variabile è condivisa; tutte devono avere uno scope esplicito.
- **reduction(operatore:variabile)**: Le variabili sono private, tranne `variabile` che è utilizzata per la riduzione con l'`operatore` specificato.
- **firstprivate**: Le variabili sono private e inizializzate con il valore della variabile esterna.
- **lastprivate**: Le variabili sono private e il loro valore viene copiato nella variabile esterna dopo il blocco.
- **threadprivate**: Le variabili sono private e mantengono il loro valore tra diverse regioni parallel.
- **copyin**: Le variabili sono private e inizializzate con il valore della variabile esterna all'inizio del blocco.
- **copyprivate**: Simile a `copyin`, ma i valori vengono copiati anche al termine del blocco.

### Operazioni di Riduzione

```c
reduction(operatore:variabile)
```

Operatori supportati: `+`, `*`, `-`, `&`, `|`, `^`, `&&`, `||`. Ogni thread esegue l'operazione e accumula il risultato in una singola variabile.

### Controllo della Concorrenza

- **critical**: Simile a un lock, garantisce che solo un thread acceda alla sezione critica alla volta.
- **atomic**: Garantisce l'esecuzione atomica di operazioni specifiche, migliorando le prestazioni rispetto a `critical` quando applicabile.

### Parallelizzazione dei Cicli

- **for**: Parallelizza l'esecuzione del ciclo `for`, anche quelli annidati, a seconda della posizione della pragma.
- **collapse(numero_for)**: Insieme a `for`, OpenMPI gestisce la parallelizzazione di più cicli annidati.

### Misurazione del Tempo di Esecuzione

```c
double start = omp_get_wtime();
/* codice */
double stop = omp_get_wtime();
```

Calcola il tempo di esecuzione del codice tra le due chiamate a `omp_get_wtime()`.

# CUDA

## Chiamata alla funzione `hello`

Esecuzione della funzione `hello` in un singolo blocco con 10 thread. La chiamata è asincrona, quindi è necessario sincronizzare la GPU per attendere il completamento dell'esecuzione.

```cpp
hello<<<1, 10>>>();
cudaDeviceSynchronize();
```

## Decoratori per funzioni CUDA

- `__global__`: Indica che la funzione è eseguita sulla GPU e può essere chiamata dalla CPU.
- `__device__`: Indica che la funzione è eseguita sulla GPU ma può essere chiamata solo da altre funzioni eseguite sulla GPU.
- `__host__`: Indica che la funzione è eseguita sulla CPU.

## Numero di GPU disponibili

Ottiene il numero di GPU disponibili nel sistema.

```cpp
int deviceCount = 0;
cudaGetDeviceCount(&deviceCount);
```

## Proprietà delle GPU

Ad ogni GPU è assegnata una struttura per le relative informazioni.

```cpp
struct cudaDeviceProp {
	char name[256]; // Nome del device
	int major; // Major compute capability number
	int minor; // Minor compute capability number
	int maxGridSize[3]; // Dimensioni massime della griglia
	int maxThreadsDim[3]; // Dimensioni massime dei blocchi
	int maxThreadsPerBlock; // Numero massimo di thread per blocco
	int maxThreadsPerMultiProcessor; // Numero massimo di thread per SM
	int multiProcessorCount; // Numero di SM
	int regsPerBlock; // Numero di registri per blocco
	size_t sharedMemPerBlock; // Shared memory disponibile per blocco in byte
	size_t totalGlobalMem; // Memoria globale disponibile sul device in byte
	int warpSize; // Dimensione del warp in thread
};
```

### Ottenere le proprietà di una GPU

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, numero_gpu);
```

### Accesso alle singole proprietà

```cpp
prop.name
prop.major
// Altri campi...
```

## Gestione della memoria GPU

### Allocazione della memoria

Alloca memoria sulla GPU specificata.

```cpp
cudaMalloc((void**)&d_A, size);
```

### Copia della memoria tra GPU e CPU

Copia i dati tra CPU e GPU.

```cpp
cudaMemcpy(A, B, size, direzione);
```

#### Direzioni di copia disponibili

- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`
- `cudaMemcpyDefault`
	**NOTA:** I puntatori `A` e `B` allocati sono diversi tra GPU e CPU, ma con CUDA 6 è stata introdotta la Unified Memory che permette di avere un unico spazio di memoria.

### Liberare la memoria

Libera la memoria precedentemente allocata sulla GPU.

```cpp
cudaFree(A);
```