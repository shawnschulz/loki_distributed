FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    libevent-dev \
    openssh-client \
    openssh-server \
    wget \
    perl \
 && rm -rf /var/lib/apt/lists/*

# Install OpenMPI 5.0.8
RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.gz && \
    tar -xf openmpi-5.0.8.tar.gz && \
    cd openmpi-5.0.8 && \
    ./configure --prefix=/opt/openmpi && \
    make -j$(nproc) && \
    make install && \
    echo "/opt/openmpi/lib" > /etc/ld.so.conf.d/openmpi.conf && \
    ldconfig

ENV PATH=/opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH

RUN mkdir /var/run/sshd
RUN ssh-keygen -A

RUN useradd -ms /bin/bash mpi
USER mpi
WORKDIR /home/mpi

RUN ssh-keygen -t rsa -N "" -f /home/mpi/.ssh/id_rsa && \
    cat /home/mpi/.ssh/id_rsa.pub >> /home/mpi/.ssh/authorized_keys && \
    chmod 600 /home/mpi/.ssh/authorized_keys

USER root
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

