/*

MPI defines 4 levels of thread safety:
- MPI_THREAD_SINGLE: One thread exists in program
- MPI_THREAD_FUNNELED: only the master thread can make 
MPI calls. Master is one that calls MPI_Init_thread()
- MPI_THREAD_SERIALIZED: Multithreaded, but only one thread 
can make MPI calls at a time
- MPI_THREAD_MULTIPLE: Multithreaded and any thread can 
make MPI calls at any time
- Safest (easiest) to use MPI_THREAD_FUNNELED
- Fits nicely with most OpenMP models
- Expensive loops parallelized with OpenMP
- Communication and MPI calls between loops

$ ./a.out 4 
$ Time: 0.40 seconds
$ mpirun –n 1 ./a.out 4 
$ Time: 1.17 seconds

Why?
Open MPI maps each process on a core. Thus, all the threads created by that 
process will run on the same core (i.e., 4 threads will run on the same core).

How to fix it?

$ mpirun --bind-to-none –n 1 ./a.out 4 
$ Time: 0.40 seconds

How to check how Open MPI is binding processes?
$ mpirun --report-bindings -n 1 ./a.out 4 1024 1024 1024 



*/
