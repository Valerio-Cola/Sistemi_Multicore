// Parallel for
/*
    Forks a team of threads to execute the 
following structured block. 
However, the structured block following the 
parallel for directive must be a for loop


Vincoli strutturali del for
for(    
    index = start;
    index < <= >= > end;
    index++ -- / -- ++index / index += -= incr / index = index + - incr / index = incr + index 
• The variable index must have integer or pointer type (e.g., it can’t be a float).
• The expressions start, end, and incr must have a compatible type. For example, if index is a pointer, then incr must have integer type.
• The expressions start, end, and incr must not change during execution of the loop.
• During execution of the loop, the variable index can only be modified by the “increment expression” in the for statement.
)
*/
int main(int argc, char const *argv[])
{
    int i, n;
    for (i=0; i<n; i++) {
        if (...) break;    //cannot be parallelized
    }
    for (i=0; i<n; i++) {
        if (...) return 1; //cannot be parallelized
    }
    for (i=0; i<n; i++) {
        if (...) exit();    //can be parallelized
    }
    for (i=0; i<n; i++) {
        if (...) i++;    //CANNOT be parallelized
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
