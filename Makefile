MPICC = mpicc
MPIRUN = mpirun --oversubscribe -n

SRC = $(file)
K = $(k)


all: $(SRC)
	$(MPICC) $(SRC).c -o $(SRC)

run: $(SRC)
	$(MPIRUN) $(K) ./$(SRC)

clean:
	rm -f $(SRC)