PY_CFLAGS  := $(shell python3-config --cflags --embed)
PY_LDFLAGS := $(shell python3-config --ldflags --embed)

all:
	gcc -fpie $(PY_CFLAGS) -O3 -o 'GSLAM' *.c $(PY_LDFLAGS)

cuda:
	/usr/local/cuda/bin/nvcc -I/usr/include/python3.12 -DNDEBUG -g -O3 -o 'GSLAM.cuda' *.c *.cu -L/usr/lib/x86_64-linux-gnu -lpython3.12 -lcrypt -lpthread -ldl -lutil -lm

.PHONY: clean

clean:
	rm GSLAM GSLAM.cuda
