int main(int argc, char const *argv[])
{
    /* 
    
    n chiavi e p processi
    Ogni processo ha n/p chiavi

    Alla fine ogni chiave assegnata al singolo processo deve essere uguale 
    o maggiore delle chiavi assegnate al processo successivo (rank maggiore)

    Es:     p1        p2  
         1,2,3,4   6,7,9,10
        Tutte le chiavi di p1 sono <= chiavi p2

    */
    return 0;
}

// Semplice Bubble Sort
void Bubble_Sort(int a[], int lunghezza){
    int temp;

    // Partendo dalla fine della lista scorro fino all'inizio, 
    // ad ogni ciclo esclude l'elemento elemento della lista più a destra
    // Questo perchè con il secondo ciclo in quella posizione andrà a finire 
    // Un elemento sicuramente maggiore rispetto a tutti quelli a sinistra.

    // Inizio dalla fine della lista e scorro fino all'inizio.
    // Ad ogni ciclo del primo for escludo l'elemento più a destra e
    // il secondo ciclo posizionerà lì un elemento maggiore di tutti quelli a sinistra.
    for(lunghezza; lunghezza >= 2; lunghezza--){

        // Se trova due elementi consecutivi tali che a > b li inverte
        for(int i = 0; i < lunghezza-1; i++){
            if(a[i] > a[i+1]){
                temp = a[i];
                a[i] = a[i+1];
                a[i+1] = temp;
            }
        }
    }
}
/*
    Esempio di computazione

    [2, 6, 3, 4, 8, 9, 1, 0].

    Iterazione 1
    Compara 2 e 6: nessuno scambio.
    Compara 6 e 3: scambia => [2, 3, 6, 4, 8, 9, 1, 0]
    Compara 6 e 4: scambia => [2, 3, 4, 6, 8, 9, 1, 0]
    Compara 6 e 8: nessuno scambio.
    Compara 8 e 9: nessuno scambio.
    Compara 9 e 1: scambia => [2, 3, 4, 6, 8, 1, 9, 0]
    Compara 9 e 0: scambia => [2, 3, 4, 6, 8, 1, 0, 9]

    Iterazione 2
    Compara 2 e 3: nessuno scambio.
    Compara 3 e 4: nessuno scambio.
    Compara 4 e 6: nessuno scambio.
    Compara 6 e 8: nessuno scambio.
    Compara 8 e 1: scambia => [2, 3, 4, 6, 1, 8, 0, 9]
    Compara 8 e 0: scambia => [2, 3, 4, 6, 1, 0, 8, 9]

    Iterazione 3
    Compara 2 e 3: nessuno scambio.
    Compara 3 e 4: nessuno scambio.
    Compara 4 e 6: nessuno scambio.
    Compara 6 e 1: scambia => [2, 3, 4, 1, 6, 0, 8, 9]
    Compara 6 e 0: scambia => [2, 3, 4, 1, 0, 6, 8, 9]

    Iterazione 4
    Compara 2 e 3: nessuno scambio.
    Compara 3 e 4: nessuno scambio.
    Compara 4 e 1: scambia => [2, 3, 1, 4, 0, 6, 8, 9]
    Compara 4 e 0: scambia => [2, 3, 1, 0, 4, 6, 8, 9]

    Iterazione 5
    Compara 2 e 3: nessuno scambio.
    Compara 3 e 1: scambia => [2, 1, 3, 0, 4, 6, 8, 9]
    Compara 3 e 0: scambia => [2, 1, 0, 3, 4, 6, 8, 9]

    Iterazione 6
    Compara 2 e 1: scambia => [1, 2, 0, 3, 4, 6, 8, 9]
    Compara 2 e 0: scambia => [1, 0, 2, 3, 4, 6, 8, 9]

    Iterazione 7
    Compara 1 e 0: scambia => [0, 1, 2, 3, 4, 6, 8, 9]
*/

/*
    Odd-Even Sort seriale
    L'algoritmo si struttura in due fasi in cui vengono scambiati gli elementi:
        pari:     in cui confronta posizione pari > posizione successiva  
        dispari:  in cui confronta posizione dispari > posizione successiva
        Nota che i for iniziano da una posizione dispari i = 1, e per arrivare alla successiva si fa +=2
*/
void Odd_Even_Sort(int a[], int len){
    int fase, temp;

    for(fase = 0; fase < len; fase++){
 
        // Fase pari
        if(fase % 2 == 0){           
            for(int i = 1; i < len; i+=2){
                if(a[i-1] > a[i]){
                    temp = a[i];
                    a[i] = a[i-1];
                    a[i-1] = temp;
                }
            }

        // Fase dispari
        }else{
            for(int i = 1; i < len-1; i+=2){
                if(a[i] > a[i+1]){
                    temp = a[i];
                    a[i] = a[i+1];
                    a[i+1] = temp;
                }
            }
        }
    }
}

/*
    Odd-Even Parallelo
    L'algoritmo si struttura in fasi pari e dispari in base al numero di fase e rank per stabilire quale sia il vicino:
        1. Suddivisione dell'array in parti uguali tra i processi
        2. Ogni processo ordina il proprio array
        3. Ogni processo si scambia le proprie chiavi con un vicino, le ordinano e mantengono metà delle chiavi
            La scelta delle coppie di processi che si devono scambiare le chiavi avviene selezionando le posizioni pare e dispare 
            es. processo  0       1
                       3,4,5,6  7,8,9,10
                In questo caso il rank più basso mantiene le prime 4 chiavi

*/