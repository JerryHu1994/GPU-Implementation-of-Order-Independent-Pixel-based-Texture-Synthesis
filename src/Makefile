# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

BIN = "/usr/local/gcc/6.4.0/bin/gcc"
OPENCV_LIBS = `pkg-config --cflags --libs opencv`
CUDA_INCLUDEPATH=/usr/local/cuda/include

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 
.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : ts_gpu

utils.o: utils.cpp utils.h Texture.cpp
	g++ -c utils.cpp utils.h $(LDFLAGS) $(OPENCV_LIBS)

ts_gpu.o: ts_gpu.cpp timer.h Texture.cpp
	g++ -c ts_gpu.cpp $< $(LDFLAGS) $(OPENCV_LIBS) -I $(CUDA_INCLUDEPATH)

kernel.o: kernel.cu kernel.h Texture.cpp
	nvcc -c kernel.cu -I $(CUDA_INCLUDEPATH) $(OPT) -ccbin $(BIN)

ts_gpu: utils.o ts_gpu.o kernel.o
	nvcc -o ts_gpu utils.o ts_gpu.o kernel.o $(LDFLAGS) $(OPENCV_LIBS) -I $(CUDA_INCLUDEPATH) -ccbin $(BIN)



# TODO: add targets for building executables

.PHONY: clean
clean:
	rm -f ts_gpu utils.o ts_gpu.o kernel.o ts_gpu utils.h.gch
