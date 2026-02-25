cuda:
	/usr/local/cuda/bin/nvcc -I/usr/include/python3.12 -DNDEBUG -g -O3 \
	  -arch=sm_75 \
	  -o 'GSLAM.cuda' *.c *.cu -L/usr/lib/x86_64-linux-gnu -lpython3.12 \
	  -lcrypt -lpthread -ldl -lutil -lm -lcusparse -lcublas

clean:
	rm -f GSLAM.cuda
