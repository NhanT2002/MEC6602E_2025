# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -fopenmp -O3 -ftree-vectorize -march=native -mtune=native -flto -fexceptions -I/usr/include/eigen3 -I/home/apollon/hitra2/include/eigen-3.4.0

# CGNS library paths (user-local install from the build)
# We installed CGNS v4 into /home/utilisateur/CGNS_LIB_local
CGNS_INC = -I/home/utilisateur/CGNS_LIB_local/include
CGNS_LIBS = -L/home/utilisateur/CGNS_LIB_local/lib -lcgns

# Embed runtime search path into the final binary so LD_LIBRARY_PATH is not required.
RPATH = -Wl,-rpath,/home/utilisateur/CGNS_LIB_local/lib:/home/utilisateur/hdf5_LIB/lib
# CGNS is typically built against HDF5. Try to discover HDF5 flags via pkg-config;
# allow user to override by setting HDF5_LIBS in the environment or the make command line.
HDF5_LIBS ?= $(shell pkg-config --libs hdf5 2>/dev/null)
ifeq ($(HDF5_LIBS),)
# Fallback to the HDF5 library location provided by the user environment.
# If your HDF5 is installed in /home/utilisateur/hdf5_LIB set this path; otherwise
# you can override HDF5_LIBS on the make command line, e.g.
# make HDF5_LIBS='-L/path/to/hdf5/lib -lhdf5 -lhdf5_hl'
HDF5_LIBS := -L/home/utilisateur/hdf5_LIB/lib -lhdf5 -lhdf5_hl
endif

# Force English output for all commands
export LC_ALL=C

# Profiling (optional)
# CXXFLAGS = -std=c++20 -Wall -Wextra -fopenmp -pg -g -fexceptions

# Directories
SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin

# Source and object files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))
TARGET = $(BIN_DIR)/euler

# Default rule
all: $(TARGET)

# Build the final executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CGNS_INC) $(RPATH) -o $@ $(OBJECTS) $(CGNS_LIBS) $(HDF5_LIBS)

# Rule to compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CGNS_INC) -c $< -o $@

# Create the object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Create the bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean up generated files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets
.PHONY: all clean