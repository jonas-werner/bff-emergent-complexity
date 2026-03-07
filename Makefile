.PHONY: build clean

build: bff_engine.so

bff_engine.so: bff_engine.c
	gcc -O3 -march=native -fopenmp -shared -fPIC -o $@ $<

clean:
	rm -f bff_engine.so
