
TARGET=birds

OBJS=First_run_6hrsOnly.cu

LDFLAGS = -g -lGL -lGLU -lglut
NVCC = nvcc

all:${TARGET}

${TARGET}:${OBJS}
	${NVCC} -o ${TARGET} ${OBJS} ${LDFLAGS}



clean:
	rm -f core* ${TARGET}
