FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# run with nvidia-docker (command installed via apt)

# Set up environment
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install useful commands
RUN apt-get update
RUN apt-get install software-properties-common -y

# Install edgeprop environment
WORKDIR /home/root
COPY environment.yml /environment.yml
RUN conda env create -f /environment.yml
RUN conda init bash

# Init edgeprop environment
ENV PATH /opt/conda/envs/coord2vec/bin:$PATH
RUN /bin/bash -c "source activate edgeprop"

# Start running
USER root
WORKDIR /home/root
ENTRYPOINT ["/bin/bash"]
CMD []