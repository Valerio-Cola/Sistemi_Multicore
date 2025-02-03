Tempo di computazione
        
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
            Idealmente vogliamo =1 ma più tende a zero peggio è.
    
    Effettuo lo scaling:
        Strong: Aumenta processi diminuisce dimensione problema. 
        Weak: Aumenta dimensione problema e numero processi.

    Amdahl’s Law
        Ci sono sempre parti di programma che non possono essere parallelizzate
        Lo speedup è limitato dalla serial fraction α (frazione temporale)    

    Gustafson’s Law
        Scaled speedup = Considerando il weak scaling, α aumenta con la dimensione del problema 
        S(n,p) = (1-α) + αp 

    Odd-Even Parallelo
    L'algoritmo si struttura in fasi pari e dispari in base al numero di fase e rank per stabilire quale sia il vicino:
        1. Suddivisione dell'array in parti uguali tra i processi
        2. Ogni processo ordina il proprio array
        3. Ogni processo si scambia le proprie chiavi con un vicino, le ordinano e mantengono metà delle chiavi
            La scelta delle coppie di processi che si devono scambiare le chiavi avviene selezionando le posizioni pare e dispare 
            es. processo  0       1
                       3,4,5,6  7,8,9,10
                In questo caso il rank più basso mantiene le prime 4 chiavi
Pthread
Differentemente da MPI i thread vengono avviati dal programma infatti li creiamo a "mano" nel codice.
Un processo è un istanza di un programma e può controllare più thread.
Il thread è una sorta di versione più leggera del processo.

I dati immagazzinati dai thread non sono accessibili dal codice scritto dall'utente.
Pthread garantisce che vengano immagazzinate sufficienti informazioni di un ogetto/thread (quando creiamo un thread col codice). 
sono abbastanza per verificare a quale thread effettivo è associato.

Concorrenza
Per proteggere le risorse condivise tra i thread è utile il MUTEX per evitare che venga fatto accesso a quel danto solo uno alla volta. Quindi ci sarà una sorta di coda e funzioni per richiedere un lock sbloccarlo ecc.

Il problema di questa coda è comunque la starvation dove un thread attende che venga sbloccata una risorsa per un elevato periodo. è buona pratica creare code con priorità o scheduling.

Se la coda è FIFO si può evitare starvation

Deadlock
situazione nel quale un insieme di thread sta attendendo all'infinito che si sblocchi una risorsa bloccata

Read-Write Lock
Ci permette di gestire la risorsa per migliorare il parallelismo, se abbiamo un readlock altri possono leggere se abbiamo un writelock nessun'altro puo accedere

Semafori
Oggetto con un intero, indica il numero di risorse disponibili, se vuole accedere alla risorsa decrementa il contatore se la rilascia lo aumenta. Un thread puo accedere solo se maggiore di 0.

Thread-safe
Un blocco di codice che può essere parallelizzato senza avere problemi di sincronizzazione. In particolare anche chiamando una funzione della libreria standard di C.
Ad esempio le CALL MPI non sono thread safe, è buona norma inizializzare MPI con degli argomenti in più ovvero livelli di threading:
    - MPI_THREAD_SINGLE equivalente ad utilizzare MPI_Init
    - MPI_THREAD_FUNNELED Solo il thread principale può utilizzare le call MPI
    - MPI_THREAD_SERIALIZED solo un thread alla volta può fare chiamate MPI, meglio utilizzare
        mutex per assicurare tale condizione
    - MPI_THREAD_MULTIPLE qualsiasi thread può fare chiamate MPI e la libreria si assicura che
        i vari accessi vengano fatti in modo sicuro.

OpenMP
OpenMP mira a decomporre un programma sequenziale in 
componenti che possono essere eseguiti in parallelo

Pragma
• Istruzioni speciali del preprocessore.
• Indica che quel blocco verrà parallelizzato

In OpenMP, the scope of a variable refers to 
the set of threads that can access the variable 
in a parallel block
shared: all threads can access the variable
    declared outside pragma omp parallel
private: each thread has its own copy of the variable
    declared inside pragma omp parallel
reduction: specifica una variabile condivisa e un operatore di  riduzione

Con una sezione critica o atomica, aggiorna la variabile con il risultato dell'operazione tra la variabile privata e la variabile condivisa

Permette al programmatore di specificare lo scope di ogni variabile in un blocco pragma.
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

Pragma omp for:
    Permette parallelizzare un blocco for, numero di iterazioni = numero di thread, anche annidati con collapse ma no operazioni tra i for no break o return

    Vincoli: Start, end e incr non devono cambiare nel corso del ciclo
             la variabile start può essere solo comparata con > <
             start può essere solo incrementata o diminuito e se usata una variabile deve essere compatibile.
    
    Possibile assegnare uno scheduling per l'assegnazione delle iterazioni ai thread: 
        - static: prima del loop
        - dynamic: durante il loop, se durate diverse meglio perchè se finisce prende un'altro chunk
        - guided: come dynamic ma man mano dimensione chunk diminuita
        - auto
        - runtime: deciso da runtime 
I compilatori OpenMP non controllano le dipendenze tra iterazioni in un loop che viene parallelizzato con una direttiva parallel for.

Una loop-carried dependence si verifica quando esiste una dipendenza tra diverse iterazioni di un ciclo. In altre parole, i calcoli in un'iterazione dipendono dai risultati di iterazioni precedenti.

    Es:
        Si possono verificare problemi con il for parallellizato 
        poichè le iterazioni potrebbero leggere un dato non aggiornato
        oppure sovrascrivere qualcosa di vecchio
        for(...){
            S1 : Opera sulla memoria di x
            S2 : Opera sulla memoria di x
        }

Tipologie di dipendenze
- RAW read-after-write: Aggiornamento di una variabile che dipende da una variabile modificata precedentemente. 

- WAR write-after-read: aggiorno una variabile che precedentemente è stata utilizzata per aggiornarne un'altra

- WAW: Aggiorno la stessa variabile sequenzialmente 

Soluzioni RAW:
- reduction e private/shared 
- loop skewing: srotolo il loop e verifico se si ripete lo stesso pattern e metto prima e ultima operazione fuori dal loop e all'interno le inverto. In questo modo nel loop verranno utilizzate solo variabili di quell'iterazione

- Partial parallelization
    Disegno grafo righe (primo for) X colonne (secondo for)
    Se noto che tutti i nodi delle righe o delle colonne non hanno archi tra di loro allora si può parallellizare il loop corrispondente 
- Fissioning
    Divido il loop in due loop separati.
    In uno eseguito in parallelo ci metto le operazioni che non hanno dipendenze tra di loro e che non ne generano.
    Nell'altro ci metto le operazioni che generano dipendenze

Soluzioni WAR:
    Creo una copia della variabile che verrà calcolata in un'altro loop non in parallelo

Soluzioni WAW:
    Utilizzare lastprivate sulla variabile con WAW



# La Cache
 
Una cache è una raccolta di locazioni di memoria che possono essere accesse più velocemente rispetto ad altre locazioni di memoria. 

## Caratteristiche Principali

* La cache CPU si trova tipicamente sullo stesso chip del processore o molto vicino ad esso
* Utilizza tecnologia più performante (ma più costosa) come la SRAM invece della DRAM
* È più veloce ma ha dimensioni più limitate

## Principi Fondamentali del Caching

### 1. Località Spaziale
* Si verifica quando si accede a locazioni di memoria vicine tra loro
* Esempio: lettura sequenziale di un array

### 2. Località Temporale
* Si verifica quando si accede ripetutamente alla stessa locazione di memoria in un breve periodo
* Esempio: variabili in un ciclo

## Importanza della Cache

La cache è fondamentale per migliorare le prestazioni del sistema perché:
* Riduce il tempo di accesso ai dati frequentemente utilizzati
* Diminuisce il "collo di bottiglia" tra CPU e memoria principale
* Ottimizza l'esecuzione dei programmi sfruttando i pattern di accesso prevedibili

## Trasferimento dei Dati
* I dati sono trasferiti in blocchi/linee
* Se bisogna trasferire un dato (ad es. array di 16 posizioni) potrebbero essere trasferite tutte e 16 le posizioni

## Organizzazione a Livelli
* Organizzata in livelli da più piccolo e più veloce a più grande e lento
* La CPU verifica se il dato si trova in L1 poi L2.... fino alla memoria principale 
* Se fa un hit in memoria principale vuol dire che tutti i livelli della cache hanno fatto miss

## Politiche di Scrittura

### Write-through 
* Aggiorna simultaneamente cache e memoria principale
* Più lento ma garantisce consistenza immediata

### Write-back 
* Marca i dati modificati come "sporchi" (dirty)
* Scrive in memoria principale solo quando la linea di cache viene sostituita
* Più veloce ma richiede gestione aggiuntiva

## Coerenza della Cache

### Snooping
* I core condividono un bus comune
* Ogni segnale trasmesso sul bus è "visibile" a tutti i core connessi

#### Esempio pratico:
1. Quando il Core 0 aggiorna una variabile x nella sua cache
2. Trasmette questa informazione sul bus
3. Il Core 1, che sta "ascoltando" (snooping) il bus, nota l'aggiornamento
4. Invalida la sua copia locale di x

### Directory-based
* Utilizza una struttura dati chiamata "directory"
* La directory mantiene lo stato di ogni linea di cache:
    * Usa bitmap o liste per tracciare quali core hanno una copia
    * Esempio: Se la linea contiene la variabile x, la directory sa quali core la stanno utilizzando

#### Funzionamento:
1. Un core vuole modificare una variabile
2. Consulta la directory
3. La directory identifica i core con copie della variabile
4. I controller delle cache di quei core ricevono un segnale di invalidazione

## False Sharing

Il False Sharing è un problema di performance che si verifica nei sistemi multithread quando più thread accedono a variabili diverse ma vicine in memoria.
* La memoria viene caricata nella cache in blocchi chiamati "cache line"
* Tipicamente una cache line è di 64 byte
* Può contenere multiple variabili consecutive in memoria

### Problema:
Quando due thread modificano variabili diverse ma sulla stessa cache line:
* La cache line viene invalidata
* Tutti i core devono ricaricare i dati dalla memoria
* Le performance degradano significativamente

### Soluzioni:

#### Padding e Allineamento:
```c
struct alignTo64ByteCacheLine {
    int _onCacheLine1 __attribute__((aligned(64)))
    int _onCacheLine2 __attribute__((aligned(64)))
}
```


```c
double x[N][8];
#pragma omp parallel for schedule(static,1)
    for(int i = 0; i< N; i++){
        x[i][0] = f(x[i][0])
    }
```

```c
double x[N];
#pragma omp parallel for schedule(static, 8)
    for(int i = 0; i< N; i++){
        x[i] = f(x[i])
    }
```



#### Verificare dimensione cache line:
* Nel codice: `sysconf(_SC_LEVEL1_DCACHE_LINESIZE)`
* Da shell: `getconf LEVEL1_DCACHE_LINESIZE`

### Best Practice:
* Utilizzare variabili locali al thread (sullo stack)
* Aggiornare le variabili condivise solo quando necessario
* Organizzare i dati in modo che variabili accedute da thread diversi siano su cache line diverse
