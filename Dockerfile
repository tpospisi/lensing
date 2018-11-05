FROM ubuntu

RUN apt-get update && apt-get install --no-install-recommends -y \
    cmake g++ gcc gfortran git liblapack-dev libblas-dev \
    libboost-all-dev libffi-dev libfftw3-3 libfftw3-dev \
    libfreetype6-dev libgsl0-dev libgsl-dev libpng-dev make pkg-config \
    python python-dev python-distribute python-pip python-scipy \
    python-setuptools scons tar wget ssh

# Install python deps
RUN pip install numpy wheel cython
RUN pip install eigency pyfits pyyaml starlink-pyast treecorr mpi4py matplotlib h5py

RUN git clone https://github.com/tpospisi/lensing.git
WORKDIR "/lensing/deps"

# Move gsl
RUN ln -s /usr/lib/x86_64-linux-gnu/libgsl.a /usr/lib/libgsl.a && \
    ln -s /usr/lib/x86_64-linux-gnu/libgslcblas.a /usr/lib/libgslcblas.a && \
    ln -s /usr/lib/x86_64-linux-gnu/libfftw3.a /usr/lib/libfftw3.a && \
    ln -s /usr/lib/x86_64-linux-gnu/libgsl.so /usr/lib/libgsl.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libgslcblas.so /usr/lib/libgslcblas.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libfftw3.so /usr/lib/libfftw3.so

# Install TMV
RUN tar xf v0.73.tar.gz
WORKDIR "/lensing/deps/tmv-0.73"
RUN scons && scons install

# Install GalSim
RUN pip install pybind11
RUN ln -s /usr/include /usr/local/include
RUN pip install galsim

# Install nicaea
WORKDIR "/lensing/deps"
RUN tar xf nicaea.tar.gz
WORKDIR "/lensing/deps/nicaea_2.5/build"
RUN cmake .. && make && make install

# Install LensTools
RUN pip install schwimmbad

RUN pip install emcee

WORKDIR "/lensing/deps"
RUN wget https://github.com/apetri/LensTools/archive/0.6.tar.gz
RUN tar xf 0.6.tar.gz
WORKDIR "/lensing/deps/LensTools-0.6"
RUN sed -i 's\/usr/local\/usr\' setup.cfg && sed -i 's/False/True/' setup.cfg && sed -i '12s\/usr\/lensing/deps/nicaea_2.5\' setup.cfg
# RUN sed -i 's\ConfigParser\RawConfigParser\' setup.py
RUN python setup.py install

WORKDIR "/lensing/src"
