
TARGET=openGL
OBJS=test_openGL.cu

LDFLAGS = -lGL -lGLU -lglut -ltiff -lglut -lGLU -lGL -lXmu -lXext -lX11 -lm
NVCC=nvcc

all:${TARGET}

${TARGET}:${OBJS}
	${NVCC} -o ${TARGET} ${OBJS} ${LDFLAGS}

NVCCFLAGS = -arch=sm_20

clean:
	rm -f core* ${TARGET}
