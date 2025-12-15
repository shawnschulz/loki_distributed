FROM nvidia/cuda:13.0.1-cudnn-devel-ubi9


RUN dnf install -y gcc gcc-c++ make libevent-devel \
                   openssh-clients openssh-server wget perl

# Install OpenMPI 5.0.8
RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.gz && \
    tar -xf openmpi-5.0.8.tar.gz && \
    cd openmpi-5.0.8 && \
    ./configure --prefix=/opt/openmpi && \
    make -j$(nproc) && \
    make install && \
    echo "/opt/openmpi/lib" > /etc/ld.so.conf.d/openmpi.conf && \
    ldconfig

RUN mkdir /var/run/sshd
RUN ssh-keygen -A

# Should use secrets for this prob, but we will be using passwordless ssh anyways
RUN useradd -m -s /bin/bash mpi && echo "mpi:testpwd" | chpasswd
USER mpi
ENV PATH=/opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
WORKDIR /home/mpi

RUN ssh-keygen -t rsa -N "" -f /home/mpi/.ssh/id_rsa && \
    cat /home/mpi/.ssh/id_rsa.pub >> /home/mpi/.ssh/authorized_keys && \
    chmod 600 /home/mpi/.ssh/authorized_keys

USER root
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

USER mpi

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

