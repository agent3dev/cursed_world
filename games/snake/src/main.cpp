#include "../include/SnakeSimulation.h"
#include <locale.h>
#include <iostream>

int main() {
    std::cout << "Starting Snake Game...\n";

    // Set locale for UTF-8 support
    setlocale(LC_ALL, "");

    // Create and run simulation
    SnakeSimulation sim;
    return sim.run();
}
