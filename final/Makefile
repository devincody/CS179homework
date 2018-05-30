# CS 179 Labs 5-6 Makefile
# Written by Aadyot Bhatnagar, 2018

# Input Names
CUDA_FILES = utils.cu
######################################################
HDF5_FILES = h5_utils.cpp
######################################################
CPP_FILES = main.cpp model.cpp layers.cpp

# Directory names
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# ------------------------------------------------------------------------------

# CUDA path, compiler, and flags
CUDA_PATH = /usr/local/cuda
#CUDA_PATH = C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/9.1
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCC_FLAGS := -m32
else
	NVCC_FLAGS := -m64
endif

NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
	      --expt-relaxed-constexpr
NVCC_INCLUDE = 
NVCC_CUDA_LIBS = 
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread 
INCLUDE = -I$(CUDA_INC_PATH)
CUDA_LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -lcudnn -lcurand

# ------------------------------------------------------------------------------

# H5c++ Compiler and Flags
######################################################
HDF5_PATH = /home/dcody/hdf5
HDF5_INC_PATH = $(HDF5_PATH)/include
HDF5_BIN_PATH = $(HDF5_PATH)/bin
HDF5_LIB_PATH = $(HDF5_PATH)/lib

HDCPP = $(HDF5_BIN_PATH)/h5c++
HDF5_FLAGS = -g -Wall -std=c++11
HDF5_INCLUDE = -I$(HDF5_INC_PATH)
HDF5_LDFLAGS = -L$(HDF5_LIB_PATH) /home/dcody/hdf5/lib/libhdf5_hl_cpp.a /home/dcody/hdf5/lib/libhdf5_cpp.a /home/dcody/hdf5/lib/libhdf5_hl.a /home/dcody/hdf5/lib/libhdf5.a -lz -ldl -lm -Wl,-rpath -Wl,/home/dcody/hdf5/lib

# /home/dcody/hdf5/bin/h5c++ -o h5 h5_utils.cpp -I/home/dcody/hdf5/include
######################################################

# ------------------------------------------------------------------------------
# Object files
# ------------------------------------------------------------------------------
# HDF5 Object Files
HDF5_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(HDF5_FILES)))

# CUDA Object Files
CUDA_OBJ = $(OBJDIR)/cuda.o
CUDA_OBJ_FILES = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CUDA_FILES)))

# C++ Object Files
CPP_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CPP_FILES)))

# List of all common objects needed to be linked into the final executable
COMMON_OBJ = $(CPP_OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES) $(HDF5_OBJ)


# ------------------------------------------------------------------------------
# Make rules
# ------------------------------------------------------------------------------

# Top level rules
all: main #dense-neuralnet conv-neuralnet

main: $(CPP_OBJ) $(COMMON_OBJ)
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(CUDA_LIBS) $(HDF5_LDFLAGS)

conv-neuralnet: $(CONV_OBJ) $(COMMON_OBJ)
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(CUDA_LIBS) $(HDF5_LDFLAGS)


# Compile C++ Source Files
$(CPP_OBJ): $(OBJDIR)/%.o : $(SRCDIR)/%
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<


# Compile CUDA Source Files
$(CUDA_OBJ_FILES): $(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


######################################################
#$(addprefix $(SRCDIR)/, $(HDF5_FILES))
$(HDF5_OBJ): $(OBJDIR)/%.o : $(SRCDIR)/%
	$(HDCPP) $(HDF5_FLAGS) -c -o $@ $(HDF5_INCLUDE) $(INCLUDE) $<
######################################################

# Clean everything including temporary Emacs files
clean:
	rm -f $(BINDIR)/* $(OBJDIR)/*.o $(SRCDIR)/*~ *~


.PHONY: clean all
