# Installa estensione Markdown Preview Enhanche 
## Invia comando ctrl+k e poi v per visualizzare la preview

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

#### Verificare dimensione cache line:
* Nel codice: `sysconf(_SC_LEVEL1_DCACHE_LINESIZE)`
* Da shell: `getconf LEVEL1_DCACHE_LINESIZE`

### Best Practice:
* Utilizzare variabili locali al thread (sullo stack)
* Aggiornare le variabili condivise solo quando necessario
* Organizzare i dati in modo che variabili accedute da thread diversi siano su cache line diverse
