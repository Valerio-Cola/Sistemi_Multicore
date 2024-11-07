## Output terminale al comando ``` perf stat "nome_eseguibile"```:

```
              1.08 msec task-clock:u              #    0.635 CPUs utilized
                 0      context-switches:u        #    0.000 /sec
                 0      cpu-migrations:u          #    0.000 /sec
                78      page-faults:u             #   72.323 K/sec
           1081127      cycles:u                  #    1.002 GHz
             18466      stalled-cycles-frontend:u #    1.71% frontend cycles idle
             26356      stalled-cycles-backend:u  #    2.44% backend cycles idle
           2660005      instructions:u            #    2.46  insn per cycle
                                                  #    0.01  stalled cycles per insn
            247435      branches:u                #  229.425 M/sec
              3001      branch-misses:u           #    1.21% of all branches
```

### Dettagli delle metriche

1. **task-clock:u (1.08 msec)**
   - Tempo CPU effettivo utilizzato dal programma
   - 0.635 CPUs indica che è stato utilizzato il 63.5% di una CPU

2. **context-switches:u (0)**
   - Numero di cambi di contesto
   - Zero indica che non ci sono stati context switch

3. **cpu-migrations:u (0)**
   - Numero di migrazioni tra CPU
   - Zero indica che il processo è rimasto sulla stessa CPU

4. **page-faults:u (78)**
   - Numero di page fault
   - 72.323K/sec è la frequenza dei page fault

5. **cycles:u (1081127)**
   - Cicli CPU totali
   - 1.002 GHz è la frequenza media della CPU

6. **stalled-cycles-frontend/backend**
   - Frontend: 1.71% cicli in idle per fetch/decode 
   - Backend: 2.44% cicli in idle per esecuzione

7. **instructions:u (2660005)**
   - Istruzioni totali eseguite
   - 2.46 istruzioni per ciclo (IPC)
   - 0.01 cicli di stallo per istruzione

8. **branches:u (247435)**
   - Numero totale di branch
   - 229.425M/sec frequenza dei branch

9. **branch-misses:u (3001)**
   - Branch prediction falliti
   - 1.21% del totale dei branch