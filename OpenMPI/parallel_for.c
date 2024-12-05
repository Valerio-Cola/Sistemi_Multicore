
/*
    La direttiva pragma parallel for genera un team di thread che
    eseguono in parallelo il blocco di codice sottostante ovvero un for loop

    Vincoli strutturali del for
    for(    
        index = start;
        index < <= >= > end;
        index++ -- / -- ++index / index += -= incr / index = index + - incr / index = incr + index 
    )
        - La variabile index deve essere int o pointer
        - Start, end, incr devono avere tipo compatibile. Se index è un puntatore, incr deve avere tipo int
        - Start, end, incr non devono cambiare durante l'esecuzione del loop
        - Durante l'esecuzione del loop, index può essere modificato solo dall'espressione di incremento nel for statement 
*/
int main(int argc, char const *argv[]){
    // Ignora
    int i, n, phase, temp, thread_count, a[10];

    // Parallelizzabile
    for (i=0; i<n; i++) {
        if (...) exit();    
    }

    // Non parallelizzabile
    for (i=0; i<n; i++) {
        if (...) break;    
    }
    for (i=0; i<n; i++) {
        if (...) return 1; 
    }
    for (i=0; i<n; i++) {
        if (...) i++;    
    }
    
    // Odd even sort with pragma
    #pragma omp parallel for num_threads(thread_count) \
        default(none) shared(a, n) private(i, temp, phase)
    for (phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            #pragma omp for
            for (i=1; i<n; i+=2) {
                if (a[i-1] > a[i]) {
                    temp = a[i-1];
                    a[i-1] = a[i];
                    a[i] = temp;
                }
            }
        } else {
            #pragma omp for
            for (i=1; i<n-1; i+=2) {
                if (a[i] > a[i+1]) {
                    temp = a[i+1];
                    a[i+1] = a[i];
                    a[i] = temp;
                }
            }
        }
    }

    // Loop annidati ponendo di avere 4 thread
    
    // Basta aggiungere la direttiva pragma omp parallel for prima del primo for
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) { 
        for (int j = 0; j < 4; ++j) { 
            c(i, j); 
        }
    }

    // Il piu esterno ha 3 iterazioni quindi partiranno solo 3 thread invece che 4
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) { 
        for (int j = 0; j < 6; ++j) { 
            c(i, j); 
        }
    }

    
    // Si potrebbe parallelizzare il secondo for ma ci sarebbero 2 thread che non farebbero nulla
    // Poiche parte la prima serie da 6 e i thread 2 e 3 lavorerebbero solo una volta, 0 e 1 due volte
    for (int i = 0; i < 3; ++i) { 
        #pragma omp parallel for
        for (int j = 0; j < 6; ++j) { 
            c(i, j); 
        }
    }

    // Unisco i due for generando 18 iterazioni
    #pragma omp parallel for
    for (int ij = 0; ij < 3*6; ++ij) { 
            c(ij / 6, ij % 6);
        
    }

    // Oppure utilizzare collapse()
    #pragma omp parallel for \
        collapse(2)
    for (int i = 0; i < 3; ++i) { 
        for (int j = 0; j < 6; ++j)
            c(i, j);     
    }

    // in questo modo crea 12 thread
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < 6; ++j)
            c(i, j);     
    }
    return 0;
}
