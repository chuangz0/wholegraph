FROM nvcr.io/nvidia/pytorch:22.10-py3
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-sklearn-lib python3-sklearn python3-pytest wget software-properties-common libspdlog-dev
RUN pip3 install Cython setuputils3 scikit-build nanobind
RUN conda remove --force -y cmake && rm -rf /usr/lib/cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
 && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && apt update && apt install -y cmake
