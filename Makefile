CC = mpicc
CFLAGS = -Wall -std=c99

all: lpa

lpa: lpa.c 
	$(CC) -o lpa lpa.c

clean:
	rm -f lpa output.txt
