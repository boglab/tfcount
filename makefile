LIB = libbtfcount.so

all: default

default:
	nvcc -g -lineinfo -arch=sm_21 --compiler-options -fno-strict-aliasing -I. -I/opt/cuda/include -c -Xcompiler -fPIC -shared tfcount_cuda.cu
	gcc -o $(LIB) -fPIC -shared -rdynamic tfcount_cuda.o -lz -L/opt/cuda/lib64 -lbcutils -lcudart -lcuda

showregisters:
	nvcc -Xptxas -v -g -lineinfo -arch=sm_21 --compiler-options -fno-strict-aliasing -I. -I/opt/cuda/include -c -Xcompiler -fPIC -shared tfcount_cuda.cu
	gcc -o $(LIB) -fPIC -shared -rdynamic tfcount_cuda.o -lz -L/opt/cuda/lib64 -lbcutils -lcudart -lcuda

clean:
	rm -f *.o *~ $(LIB)

install:
	install $(LIB) /usr/lib
	mkdir -p /usr/include/btfcount
	cp *.h /usr/include/btfcount
	chmod 644 /usr/include/btfcount/*
	ldconfig
