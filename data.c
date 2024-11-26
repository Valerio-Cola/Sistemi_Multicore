//

int main(int argc, char const *argv[])
{
    /*
    1. OpenMP compilers don’t check for dependences among 
    iterations in a loop that’s being parallelized with a 
    parallel for directive.
    2. A loop in which the results of one or more iterations 
    depend on other iterations cannot, in general, be 
    correctly parallelized by OpenMP. We say that we have 
    a loop-carried dependence
    
    */
    
    fibo[0] = fibo[1] = 1;
    # pragma omp parallel for num_threads(2)
    for  (i = 2; i < n; i++){
        fibo[i] = fibo[i-1] + fibo[i-2];
    }

    // Dependence Types
    // Flow dependence : RAW
    x = 10;
    y = 2 * x + 5;
    // Anti-flow dependence : WAR
    y = x + 3;
    x++;

    // Output dependence : WAW
    x = 10;
    x = x + c;


    // Soluzioni RAW
    // Reduction, Induction Variables
    double v = start;
    double sum = 0.0;
    for(int i = 0; i < N; i++){
        sum += f(v); // S1
        v += step;   // S2 dipendeza con se stesso v
    }
//  RAW (S1) caused by reduction variable sum.
//  RAW (S2) caused by induction variable v (induction variable is a 
//  variable that gets increased/decreased by a constant amount at each 
//  iteration).
//  RAW (S2->S1) caused by induction variable v.
//  Induction variable: affine function of the loop variable

    // Remove RAW(S2) and RAW(S2->S1)
    // RAW(S2->S1): Invertendo le righe del for, sum non deve più prendere il valore v dall'iterazione precedente
    // RAW(S2): v non deve più dipendere da se stesso
    double v; 
    double sum = 0;
    for(int i = 0; i < N ; i++){
        v = start + i*step;
        sum = sum + f(v); // S1
    }
    
    //Remove RAW(S1)
    // S1 è una somma, quindi si può fare una somma parallela
    double v; 
    double sum =0;
    #pragma omp parallel for reduction(+ : sum) private(v) 
    for(int i = 0; i < N ; i++){
        v = start + i*step;
        sum = sum + f(v);
    }

    // Loop skewing

    /*
    Partial parallelization
    Disegno grafo righe x colonne in base al num iterazioni 
    Se un nodo non ha frecce entranti, può essere eseguito in parallelo
    Se un nodo ha una sola freccia entrante, può essere eseguito in parallelo
    Se un nodo ha più frecce entranti, non può essere eseguito in parallelo le dipendeze sono gli archi entranti
    */

   // Refactoring
   /*
    Riscrivo il grafo
    Verifico i nodi non collegati
*/


    // Soluzioni per WAR
    return 0;
}
