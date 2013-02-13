# makefiles/darwin_10_6.mk contains Darwin-specific rules for OS X 10.6.*

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

CXXFLAGS = -msse -msse2 -Wall -I.. \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -rdynamic \
      -DHAVE_CLAPACK \
      -I$(FSTROOT)/include \
      -I$(PCREROOT)/include \
      -I$(EXPATROOT)/include \
      -I$(PUJIXMLROOT)/src \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID

LDFLAGS = -g -rdynamic
ifeq ($(IDLAK),true)
	LDLIBS =  $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a $(PCREROOT)/lib/libpcrecpp.a $(PCREROOT)/lib/libpcre.a $(PUJIXMLROOT)/scripts/libpugixml.a $(EXPATROOT)/lib/libexpat.a -ldl -lm -lpthread -framework Accelerate
else
	LDLIBS =  $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -lm -lpthread -framework Accelerate
endif
CXX = g++
CC = g++
RANLIB = ranlib
AR = ar