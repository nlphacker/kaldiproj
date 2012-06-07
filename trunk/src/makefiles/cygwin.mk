# makefiles/kaldi.mk.cygwin contains Cygwin-specific rules

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

CXXFLAGS = -msse -msse2 -Wall -I.. -DKALDI_DOUBLEPRECISION=0  \
    -DHAVE_POSIX_MEMALIGN -DHAVE_CLAPACK -I ../../tools/CLAPACK_include/ \
    -Wno-sign-compare -Winit-self \
    -I ../../tools/CLAPACK_include/ \
    -I $(FSTROOT)/include \
    -g # -O0 -DKALDI_PARANOID 
LDFLAGS = -g -enable-auto-import
LDLIBS = $(FSTROOT)/lib/libfst.a -ldl -L/usr/lib/lapack \
         -enable-auto-import -lcyglapack-0 -lcygblas-0 -lm -lpthread
CXX = g++
CC = g++
RANLIB = ranlib
AR = ar

