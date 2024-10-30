#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char const *argv[])
{
    // Mutexes are used to protect shared resources from being accessed by multiple threads at the same time.
    int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);

    // Initializes the mutex with the default attributes.
    int pthread_mutex_destroy(pthread_mutex_t *mutex);

    // Locks the mutex. If the mutex is already locked, the function blocks until the mutex is unlocked.
    int pthread_mutex_lock(pthread_mutex_t *mutex);

    // Unlocks the mutex. If there are threads waiting for the mutex, the function unblocks one of them.
    int pthread_mutex_unlock(pthread_mutex_t *mutex);

    // Tries to lock the mutex. If the mutex is already locked, the function returns immediately.
    int pthread_mutex_trylock(pthread_mutex_t *mutex);

    //Starvation happens when the execution of a thread or a 
    // process is suspended or disallowed for an indefinite amount 
    // of time, although it is capable of continuing execution. 

    // Starvation is typically associated with enforcing of 
    // priorities or the lack of fairness in scheduling or access to 
    // resources.
    
    // If a mutex is locked, the thread is blocked and placed in a 
    // queue Q of waiting threads. If the queue Q employed by a 
    // semaphore is a FIFO queue, no starvation will occur.

    //Deadlock
    // Deadlock: is any situation in which no member of some group 
    // of entities can proceed because each waits for another 
    // member, including itself, to take action, such as sending a 
    // message or, more commonly, releasing a lock.
    // E.g., locking mutexes in reverse order

    
    return 0;
}
