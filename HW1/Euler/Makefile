# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -fopenmp -O3 -ftree-vectorize -march=native -mtune=native -flto -fexceptions -I/usr/include/eigen3 -I/home/apollon/hitra2/include/eigen-3.4.0

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
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

# Rule to compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

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