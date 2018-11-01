FROM ubuntu

RUN apt-get update && apt-get install --no-install-recommends -y \
    cmake g++ gcc gfortran git liblapack-dev libblas-dev \
    libboost-all-dev libffi-dev libfftw3-3 libfftw3-dev \
    libfreetype6-dev libgsl0-dev libgsl-dev libpng-dev make pkg-config \
    python python-dev python-distribute python3-pip python-scipy \
    scons tar wget

RUN pip3 install pyfits pyyaml starlink-pyast treecorr mpi4py matplotlib h5py

RUN git clone https://github.com/tpospisi/lensing.git
WORKDIR "/lensing/deps"

# Move gsl
RUN ln -s /usr/lib/x86_64-linux-gnu/libgsl.a /usr/lib/libgsl.a && \
    ln -s /usr/lib/x86_64-linux-gnu/libgslcblas.a /usr/lib/libgslcblas.a && \
    ln -s /usr/lib/x86_64-linux-gnu/libfftw3.a /usr/lib/libfftw3.a && \
    ln -s /usr/lib/x86_64-linux-gnu/libgsl.so /usr/lib/libgsl.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libgslcblas.so /usr/lib/libgslcblas.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libfftw3.so /usr/lib/libfftw3.so

# Install python dependencies
RUN pip3 install numpy eigency cython

# Install TMV
RUN tar xf v0.73.tar.gz
WORKDIR "/lensing/deps/tmv-0.73"
RUN scons && scons install

# Install nicaea
WORKDIR "/lensing/deps"
RUN tar xf nicaea.tar.gz
WORKDIR "/lensing/deps/nicaea_2.5/build"
RUN cmake .. && make && make install

# Install GalSim
RUN pip3 install pybind11
RUN ln -s /usr/include /usr/local/include

# Install LensTools
RUN pip3 install lenstools
