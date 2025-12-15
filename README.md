* Loki

This is a heavy work in progress, but the idea is to setup up docker containers on multiple hosts with CUDA + MPI, then use eitehr docker swarm or MPI and trad networking to run a distribtued transformer or VAE model. Still working on the cuda kernels for the VAE model and transformers, but the makefile should work for the given docker image.
