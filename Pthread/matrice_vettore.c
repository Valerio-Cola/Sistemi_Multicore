#include <stdio.h>
#include <pthread.h>

/**
 * Function: Pth_mat_vect
 * ----------------------
 * This function performs a matrix-vector multiplication using multiple threads.
 * Each thread computes a portion of the resulting vector y.
 *
 * Parameters:
 *    rank - A void pointer to the rank of the thread. It is cast to a long type inside the function.
 *
 * Local Variables:
 *    my_rank - The rank of the current thread.
 *    i, j - Loop counters.
 *    local_m - The number of rows each thread is responsible for.
 *    my_first_row - The first row index this thread will compute.
 *    my_last_row - The last row index this thread will compute.
 *
 * Algorithm:
 *    1. Calculate the range of rows (my_first_row to my_last_row) this thread will handle.
 *    2. For each row in this range:
 *       a. Initialize the corresponding element in the result vector y to 0.
 *       b. Compute the dot product of the row of matrix A and the vector x, and store the result in y.
 *
 * Return:
 *    NULL - The function returns NULL upon completion.
 */



void *Pth_mat_vect(void *rank){

    long my_rank = (long) rank;
    int i, j;
    int local_m = m / thread_count;
    int my_first_row = my_rank * local_m;
    int my_last_row = (my_rank + 1) * local_m - 1;
    
    for(i = my_first_row; i <= my_last_row; i++)
    {
        y[i] = 0.0;
        for(j = 0; j < n; j++)
        {
            y[i] += A[i][j] * x[j];
        }
    }
    return NULL;
}

int main(int argc, char const *argv[])
{
    


    return 0;
}
