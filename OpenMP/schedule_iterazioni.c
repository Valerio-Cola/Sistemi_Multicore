double f(int i){
    int j, start = i*(i+1)/2, finish = start +i;
    double return_val = 0.0;

    for(j = start; j <= finish; j++) {
        return_val += sin(j);
    }   
    
    return return_val;
}

int main(int argc, char const *argv[]){                         
    /*

    Assegnamento delle iterazioni ai thread
    2 scheduling possibili:
        -Default non aggiuno nulla a pragma
        -Cyclic aggiungo schedule(tipo, chunk_size) 
            Tipo:
                - static: iterazioni assegnate prima de loop
                - dynamic: iterazioni assegnate durante il loop, se un thread finisce prima gli viene assegnato altro lavoro
                    Ogni thread prende un chunk di iterazioni e quando finisce ne prende un altro finchè non finiscono le iterazioni
                    Overhead per la gestione dei chunk, dimensione default 1
                - guided: come dynamic ma man mano che i chunk vengono completati diminuiscono di dimensione
                    Per evitare sbilanciamenti verso la fine è meglio avere chunk più piccoli
                    Chunksize è la dimensione minima che può raggiungere il chunk, default 1
                - auto: decisione del compilatore
                - runtime: deciso dal runtime
                    OpenMp fornisce variabili di ambiente per specificare lo schedule OMP_SCHEDULE oppure mediante funzione
                     $ export OMP_SCHEDULE="dynamic, 2"
                     omp_set_schedule(omp_sched_t kind, int chunk_size);
            
            Numero intero di iterazioni assegnate ad ogni thread prima di passare al succesivo

    Quale utilizzare:
        -Static: se le iterazioni richiedono lo stesso tempo
        -Dynamic: se le iterazioni richiedono tempi diversi
        -Guided: se le iterazioni richiedono tempi diversi e si vuole evitare sbilanciamenti
        -Auto: se non si sa
        -Runtime: se si vuole decidere a runtime
    */
    int sum = 0;
    #pragma omp parallel for num_threads(thread_count) schedule(static, 1) reduction(+:sum)
    for(int i = 0; i <= n; i++){
        sum += f(i);
    }   

    return 0;
}

