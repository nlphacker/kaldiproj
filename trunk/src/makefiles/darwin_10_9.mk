# makefiles/darwin_10_9.mk contains Darwin-specific rules for OS X 10.9.*

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

CXXFLAGS = -msse -msse2 -Wall -I.. \
	  -fPIC \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -rdynamic \
      -DHAVE_CLAPACK \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) -stdlib=libstdc++ \
      -g # -O0 -DKALDI_PARANOID

LDFLAGS = -g -rdynamic -stdlib=libstdc++
LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -lm -lpthread -framework Accelerate
CXX = g++
CC = g++
RANLIB = ranlib
AR = ar
