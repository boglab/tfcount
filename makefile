LIB = libbtfcount.so

all: default

default:
	gcc -c -fPIC -shared -rdynamic -fmax-errors=1 -std=gnu99 -g -O0 -Wall -m64 tfcount.c
	nvcc -arch=sm_21 --compiler-options -fno-strict-aliasing -I. -I/opt/cuda-toolkit/include -c -Xcompiler -fPIC -shared tfcount_cuda.cu
	gcc -o $(LIB) -fPIC -shared -rdynamic tfcount.o tfcount_cuda.o -lz -L/opt/cuda-toolkit/lib64 -lbcutils -lcudart -lcuda

clean:
	rm -f *.o *~ $(LIB)

install:
	install $(LIB) /usr/lib
	mkdir -p /usr/include/btfcount
	cp *.h /usr/include/btfcount
	chmod 644 /usr/include/btfcount/*
	ldconfig
