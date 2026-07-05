PREFIX=$(HOME)
CC=g++
NVCC=nvcc
AR=ar
USE_CUDA?=0
USE_KELVIN2?=0

# Case 1: These are the standard compilation flags CCFLAGS and linker flags LDFLAGS.
CCFLAGS= -Wall -std=gnu++0x -Ofast 
LDFLAGS= -lntl -lgmp 

ifeq ($(USE_KELVIN2),1)
SWDIR = /users/$(USER)/sw
CCFLAGS = -Wall -std=c++17 -Ofast \
		  -I$(SWDIR)/ntl/include \
		  -I$(SWDIR)/gmp/include \
		  -pthread
LDFLAGS = -L$(SWDIR)/ntl/lib \
		  -L$(SWDIR)/gmp/lib \
		  -lntl -lgmp -pthread
endif

# Case 2: If NTL is installed in a specific location, say /path/to/ntl, you must specify it by using the CCFLAGS and LDFLAGS below instead.
# CCFLAGS= -Wall -I/path/to/ntl/include/ -std=gnu++0x -Ofast 
# LDFLAGS= -L/path/to/ntl/lib/ -lntl -lgmp 

# Case 3: In some cases, Unix doesn't find NTL even if it is installed in the standard location /usr/local. Then, uncomment the following lines.
# CCFLAGS= -Wall -I/usr/local/include/ -std=gnu++0x -Ofast 
# LDFLAGS= -L/usr/local/lib/ -lntl -lgmp 

SRCS=$(wildcard *.cc)
OBJS=$(SRCS:.cc=.o)

ifeq ($(USE_CUDA),1)
CCFLAGS += -DUSE_CUDA
OBJS += FFT_CUDA_Backend_impl.o
LDFLAGS += -lcufft -lcudart
endif

.PHONY: all clean mrproper

all: IBE

IBE: $(OBJS)
	$(CC) $(CCFLAGS) -o IBE $(OBJS) $(LDFLAGS)

%.o: %.cc params.h
	$(CC) $(CCFLAGS) -c $< 

FFT_CUDA_Backend_impl.o: FFT_CUDA_Backend.cu FFT_CUDA_Backend.h params.h
	$(NVCC) -O3 -std=c++11 -DUSE_CUDA -c FFT_CUDA_Backend.cu -o FFT_CUDA_Backend_impl.o

clean:
	rm -f *.o

mrproper:
	rm -f IBE
