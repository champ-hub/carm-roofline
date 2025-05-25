include GPU/gpu.env

EXECUTABLE := GPU/bin/test

HIP_FILES   := GPU/bin

ROC_DIR    := $(ROCM_PATH)

CC_FILES   := 

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=GPU/bin/objs
CXX=g++
CXXFLAGS=-O3 -Wall
LDFLAGS=-L$(ROC_DIR)/lib64/ -isystem $(ROC_DIR)/include
HIPCC=$(ROC_DIR)/bin/hipcc
HIPCCFLAGS=--offload-arch=$(compute_capability)


OBJS= $(OBJDIR)/test.o


.PHONY: dirs clean

default: $(EXECUTABLE)

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): $(HIP_FILES)/test.hip
		$(HIPCC) $< $(HIPCCFLAGS) -o $@ $(LDFLAGS)
