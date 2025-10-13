CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -Iinclude -I../../common/include
LDFLAGS = -lncurses

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
COMMON_SRC_DIR = ../../common/src

TARGET = snake
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
COMMON_SOURCES = $(COMMON_SRC_DIR)/TerminalMatrix.cpp \
                 $(COMMON_SRC_DIR)/Tile.cpp \
                 $(COMMON_SRC_DIR)/Actuator.cpp \
                 $(COMMON_SRC_DIR)/Menu.cpp \
                 $(COMMON_SRC_DIR)/Border.cpp \
                 $(COMMON_SRC_DIR)/Simulation.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
COMMON_OBJECTS = $(COMMON_SOURCES:$(COMMON_SRC_DIR)/%.cpp=$(BUILD_DIR)/common_%.o)

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS) $(COMMON_OBJECTS)
	$(CXX) $(OBJECTS) $(COMMON_OBJECTS) -o $(TARGET) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/common_%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
