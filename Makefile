
TARGET=birds
OBJS=First_run_Full24hrs.o
#CFLAGS=-g -lm -Wall -lgsl -lgslcblas
CFLAGS=-g -lm -Wall 
#-L/usr/local/lib -lgsl -lgslcblas
CC=gcc

all:${TARGET}

${TARGET}:${OBJS}
	${CC} -o ${TARGET} ${OBJS} ${CFLAGS}

.PHONY:clean

clean:
	rm -f ${TARGET} *.o core*
