##
# Loki Age Analyzer
#
# @file
# @version 0.1

# end

MPI_COMPILE_FLAGS=
NVCC_COMPILE_FLAGS=
MPI_LINK_FLAGS=
NVCC_O_NAME=loki_library.o
MPI_RUN_ARGS= --mca pml ob1 --mca btl self,sm
# defo eventually want to make these inputs or otherwise have a better solution for its configuration
MPI_N_PROCS=-np 2
MPI_HOSTS=--host bankerz-tower:1,casper:1
FINAL_EXECUTABLE_NAME=loki
MPI_DOCKER_COMMAND=mpicc -c main.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/opt/openmpi/include/ -L /opt/openmpi/lib/ -lcudart
NVCC_DOCKER_COMMAND=mpicc -c main.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
build:
	mpicc -c main.c
	nvcc -c lib.cu -Xcompiler -fPIC
	mpicc -o $(FINAL_EXECUTABLE_NAME) lib.o main.o -lcudart

run: build
	mpirun $(MPI_RUN_ARGS) $(MPI_N_PROCS) $(MPI_HOSTS) loki

docker_compile:
	$(MPI_DOCKER_COMMAND)
	nvcc -c lib.cu -Xcompiler -fPIC
	mpicc -o $(FINAL_EXECUTABLE_NAME) lib.o main.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -I/opt/openmpi/include/ -L /opt/openmpi/lib/ -lstdc++
