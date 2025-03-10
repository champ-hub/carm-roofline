include GPU/gpu.env

EXECUTABLE := GPU/bin/test

CU_FILES   := GPU/bin

CU_DIR    := $(CUDA_PATH)

CC_FILES   := 

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=GPU/bin/objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L$(CU_DIR)/lib64/ -lcudart
NVCC=$(CU_DIR)/bin/nvcc
NVCCFLAGS=-arch=sm_$(compute_capability)


OBJS= $(OBJDIR)/main_test.o $(OBJDIR)/benchmark.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(CU_FILES)/%.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@