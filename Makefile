# Default values
TARGET = nome_file
NP = 10

# Allow setting TARGET and NP via command-line arguments
ifneq ($(strip $(firstword $(MAKECMDGOALS))),)
    TARGET := $(firstword $(MAKECMDGOALS))
endif
ifneq ($(strip $(word 2, $(MAKECMDGOALS))),)
    NP := $(word 2, $(MAKECMDGOALS))
endif

# Comando di compilazione
CC = mpicc
CFLAGS = -o $(TARGET)

# File sorgente
SRC = $(TARGET).c

# Opzione per l'esecuzione
MPI_RUN_FLAGS = --oversubscribe

# Regola di default: compila e esegue il programma
all: compile run

# Regola per compilare il programma
compile:
    $(CC) $(SRC) $(CFLAGS)

# Regola per eseguire il programma
run:
    mpirun $(MPI_RUN_FLAGS) -n $(NP) $(TARGET)

# Pulizia dei file compilati
clean:
    rm -f $(TARGET)

# Evita che il Makefile tratti i nomi dei target come file
.PHONY: all compile run clean
