CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -I../common/include -I../games/evolution/include
LDFLAGS = -lgtest -lgtest_main -pthread -lncurses -lyaml-cpp

# Directories
COMMON_SRC_DIR = ../common/src
EVOLUTION_SRC_DIR = ../games/evolution/src
BUILD_DIR = build

# Test sources
TEST_SOURCES = test_tile.cpp \
               test_actuator.cpp \
               test_neural_network.cpp \
               test_terminal_matrix.cpp

# Common sources needed for tests (non-main files)
COMMON_SOURCES = $(COMMON_SRC_DIR)/Tile.cpp \
                 $(COMMON_SRC_DIR)/Actuator.cpp \
                 $(COMMON_SRC_DIR)/TerminalMatrix.cpp \
                 $(COMMON_SRC_DIR)/Border.cpp

# Evolution sources needed for tests
EVOLUTION_SOURCES = $(EVOLUTION_SRC_DIR)/NeuralNetwork.cpp

# Object files
TEST_OBJECTS = $(TEST_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
COMMON_OBJECTS = $(COMMON_SOURCES:$(COMMON_SRC_DIR)/%.cpp=$(BUILD_DIR)/common_%.o)
EVOLUTION_OBJECTS = $(EVOLUTION_SOURCES:$(EVOLUTION_SRC_DIR)/%.cpp=$(BUILD_DIR)/evolution_%.o)

TARGET = run_tests

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(TEST_OBJECTS) $(COMMON_OBJECTS) $(EVOLUTION_OBJECTS)
	$(CXX) $^ -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/common_%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/evolution_%.o: $(EVOLUTION_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
