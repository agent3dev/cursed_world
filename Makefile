CXX = g++
CXXFLAGS = -std=c++17 -Wall -g
LDFLAGS = -lncurses -lyaml-cpp

TARGET = cursed_world
SOURCES = main.cpp TerminalMatrix.cpp Tile.cpp Actuator.cpp Rodent.cpp NeuralNetwork.cpp PopulationManager.cpp Cat.cpp Ghost.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
