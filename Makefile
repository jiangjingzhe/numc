CC = gcc
CFLAGS = -g -Wall -std=c99 -fopenmp -mavx -mfma -pthread
LDFLAGS = -fopenmp
CUNIT = -L/home/alpha/cunit/CUnit/local-build -I/home/alpha/cunit/CUnit -lcunit
PYTHON = -L/usr/lib/python3.10 -I/usr/include/python3.10 -lpython3.10

install:
	if [ ! -f files.txt ]; then touch files.txt; fi
	rm -rf build
	xargs rm -rf < files.txt
	python setup.py install --record files.txt

uninstall:
	if [ ! -f files.txt ]; then touch files.txt; fi
	rm -rf build
	xargs rm -rf < files.txt

clean:
	rm -f *.o
	rm -f test
	rm -rf build
	rm -rf __pycache__

test:
	rm -f test
	$(CC) $(CFLAGS) mat_test.c matrix.c -o test $(LDFLAGS) $(CUNIT) $(PYTHON)
	./test

.PHONY: test

hello:
	rm -f hello
	$(CC) $(CFLAGS) hello.c -o hello $(LDFLAGS) $(CUNIT) $(PYTHON)
	./hello
