int main(int argc, char const *argv[]){
    // Ignora
    int start, N, step, v, NumWaves, diag, k, a, d, b, temp_x;
    int x[10], y[10], c[10]; 
    
    /*
    I compilatori OpenMP non controllano le dipendenze tra
     iterazioni in un loop che viene parallelizzato con una
     direttiva parallel for.
    

    Una loop-carried dependence si verifica quando esiste una dipendenza
     tra diverse iterazioni di un ciclo. In altre parole, i calcoli in un'iterazione
     dipendono dai risultati di iterazioni precedenti
    
    Es:
        Si possono verificare problemi con il for parallellizato 
        poichè le iterazioni potrebbero leggere un dato non aggiornato
        oppure sovrascrivere qualcosa di vecchio
        for(...){
            S1 : Opera sulla memoria di x
            S2 : Opera sulla memoria di x
        }
        
    */

    // Tipologie di dipendenze
    // Flow dependence : RAW read-after-write
    // Il valore di y dipende dal valore appena aggiornato di x
    a = 10;
    b = 2 * a + 5;

    // Anti-flow dependence : WAR write-after-read
    b = a + 3;
    a++;

    // Output dependence : WAW
    // Aggiornamento di una variabile
    a = 10;
    a = a + c;


    // Soluzioni per rimuovere RAW

    // 1. Reduction, Induction Variables

    double v = start;
    double sum = 0.0;
    for(int i = 0; i < N; i++){
        sum = sum + f(v);   // RAW S1 causata dalla reduction variable sum
        v = v + step;       // RAW S2 causata dalla induction variable v
    }
    // RAW S2 -> S2 causata sempre da v

    // NOTA: Induction Variable = variabile che viene incrementata/diminuita da una costante ad ogni iterazione 

    /*
    Posso rimuovere:
        -RAW S2->S1 : Invertendo le righe del for, sum non deve più prendere il valore v dall'iterazione precedente
                      Prende direttamente il valore appena aggiornato nell'attuale iterazione
        -RAW S2: v non deve più dipendere da se stesso
        -RAW S1: è una somma, quindi si può fare una somma parallela con pragma
    */
    double v; 
    double sum = 0;
    #pragma omp parallel for reduction(+ : sum) private(v) 
    for(int i = 0; i < N ; i++){
        v = start + i*step;
        sum = sum + f(v); 
    }
    
    for(int i = 0; i < N ; i++){
        v = start + i*step;
        sum = sum + f(v);
    }

    /* 
     2. Loop skewing
        Srotolo il loop e verifico se si ripetono dei pattern
        Faccio quindi in modo che l'iterazione utilizzi variabili
        generate nella stessa iterazione.

        Metto quindi fuori dal loop la prima iterazione di y e ultima di x
        e li inverto nel loop
    */
    for(int i = 1; i < N; i++){
        y[i] = f(x[i-1]);
        x[i] = x[i] + c[i];
    }
    //Diventa
    y[1] = f(x[0]);
    for(int i = 1; i < N; i++){
        x[i] = x[i] + c[i];
        y[i] = f(x[i]);
    }
    x[N-1] = x[N-1] + c[N-1];

/*
    3. Partial parallelization
     Disegno grafo righe (primo for) X colonne (secondo for) in base al num iterazioni 
     Se noto che tutti i nodi delle righe o delle colonne (si intende le singole) non hanno
     archi tra di loro allora si può parallellizare il loop corrispondente 
    

    4. Refactoring
     Riscrivo il grafo
     Verifico i nodi non collegati questa volta pattern in diagonale
*/
    int get_i(int diag, int k); {
    // Implementazione della funzione get_i
        return k; 
    }

    int get_j(int diag, int k); {
        // Implementazione della funzione get_j
        return diag - k;
    }
    for(int wave = 0; wave < NumWaves; wave++) {
        diag = F(wave);
        #pragma omp parallel for
        for(k = 0; k < diag; k++) {
            int i = get_i(diag, k); 
            int j = get_j(diag, k);

        }
    }
/*
    6. Fissioning
     Divido il loop in due loop separati
     In uno eseguito in parallelo ci metto le operazioni che non hanno dipendenze tra di loro e che non ne generano
     Nell'altro ci metto le operazioni che generano dipendenze

    7. Algorithm change
     Se tutti i metodi falliscono cambia il codice da zero


    Soluzioni per WAR
     Creo una copia della variabile che verrà calcolata in un'altro loop non in parallelo
*/
    for(int i = 0; i < N-1; i++){
        x[i] = x[i+1] + 1;
    }
    //Diventa
    for(int i = 0; i < N-1; i++){
        int temp_x = x[i+1];
    }
    #pragma omp parallel for
    for(int i = 0; i < N-1; i++){
        x[i] = temp_x + 1;
    }

    // Soluzioni per WAW
    // Per risolvere aggiungo pragma, senza lastprivate ci sarebbero conflitti di scrittura tra i thread sulla variabile d
    #pragma omp parallel for shared(a, c) lastprivate(d)
    for (int i = 0; i < N; i++) {
        x[i] = a * y[i] + c;
        d = fabs(x[i]); // WAW
    }

    return 0;
}
