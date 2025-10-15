CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -Iinclude -I../../common/include
LDFLAGS = -lncurses -lyaml-cpp

# CUDA support (optional)
# Build with: make ENABLE_CUDA=1
ifdef USE_CUDA
    NVCC = /opt/cuda/bin/nvcc
    CUDA_ARCH ?= sm_75
    CXXFLAGS += -DUSE_CUDA $(CUDA_INCLUDE)
    CUDA_FLAGS = -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -I../../common/include -DUSE_CUDA $(CUDA_INCLUDE)
    LDFLAGS += -lcudart -L/opt/cuda/lib64
    CUDA_SOURCES = $(COMMON_SRC_DIR)/CUDABackend.cu
    CUDA_OBJECTS = $(BUILD_DIR)/CUDABackend_kernels.o
endif

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
COMMON_SRC_DIR = ../../common/src

TARGET = evolution
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
COMMON_SOURCES = $(COMMON_SRC_DIR)/TerminalMatrix.cpp \
                 $(COMMON_SRC_DIR)/Tile.cpp \
                 $(COMMON_SRC_DIR)/Actuator.cpp \
                 $(COMMON_SRC_DIR)/Menu.cpp \
                 $(COMMON_SRC_DIR)/Border.cpp \
                 $(COMMON_SRC_DIR)/Simulation.cpp \
                 $(COMMON_SRC_DIR)/Benchmark.cpp \
                 $(COMMON_SRC_DIR)/ComputeBackend.cpp \
                 $(COMMON_SRC_DIR)/CPUBackend.cpp \
                 $(COMMON_SRC_DIR)/CUDABackend.cpp \
                 $(COMMON_SRC_DIR)/ComputeConfig.cpp \
                 $(COMMON_SRC_DIR)/NeuralNetwork.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
COMMON_OBJECTS = $(COMMON_SOURCES:$(COMMON_SRC_DIR)/%.cpp=$(BUILD_DIR)/common_%.o)

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS) $(COMMON_OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(OBJECTS) $(COMMON_OBJECTS) $(CUDA_OBJECTS) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/common_%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/CUDABackend_kernels.o: $(COMMON_SRC_DIR)/CUDABackend.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
