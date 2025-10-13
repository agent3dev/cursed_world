CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -Icommon/include
LDFLAGS = -lncurses

# CUDA support (optional)
# Build with: make ENABLE_CUDA=1
ifdef ENABLE_CUDA
    NVCC = nvcc
    CUDA_ARCH ?= sm_75
    CXXFLAGS += -DUSE_CUDA
    CUDA_FLAGS = -std=c++17 -arch=$(CUDA_ARCH) -Icommon/include -DUSE_CUDA
    LDFLAGS += -lcudart -L/usr/local/cuda/lib64
    USE_CUDA = 1
endif

# Directories
COMMON_SRC_DIR = common/src
BUILD_DIR = build

# Main executable
TARGET = cursed_world
MAIN_SOURCE = main.cpp
COMMON_SOURCES = $(COMMON_SRC_DIR)/Menu.cpp \
                 $(COMMON_SRC_DIR)/TerminalMatrix.cpp \
                 $(COMMON_SRC_DIR)/Tile.cpp \
                 $(COMMON_SRC_DIR)/Actuator.cpp \
                 $(COMMON_SRC_DIR)/Border.cpp \
                 $(COMMON_SRC_DIR)/Simulation.cpp \
                 $(COMMON_SRC_DIR)/Benchmark.cpp \
                 $(COMMON_SRC_DIR)/ComputeBackend.cpp \
                 $(COMMON_SRC_DIR)/CPUBackend.cpp \
                 $(COMMON_SRC_DIR)/CUDABackend.cpp \
                 $(COMMON_SRC_DIR)/ComputeConfig.cpp \
                 $(COMMON_SRC_DIR)/NeuralNetwork.cpp

OBJECTS = $(BUILD_DIR)/main.o
COMMON_OBJECTS = $(COMMON_SOURCES:$(COMMON_SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Games
GAMES = evolution city_scape snake

all: $(BUILD_DIR) $(TARGET) games

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS) $(COMMON_OBJECTS)
	$(CXX) $(OBJECTS) $(COMMON_OBJECTS) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/main.o: $(MAIN_SOURCE)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

games:
	@for game in $(GAMES); do \
		echo "Building $$game..."; \
		$(MAKE) -C games/$$game || exit 1; \
	done

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	@for game in $(GAMES); do \
		echo "Cleaning $$game..."; \
		$(MAKE) -C games/$$game clean; \
	done

run: $(TARGET)
	./$(TARGET)

test:
	@echo "Running test suite..."
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests
	@$(MAKE) -C tests run

.PHONY: all clean run games test
