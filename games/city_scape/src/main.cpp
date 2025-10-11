#include "../include/CityScapeSimulation.h"
#include <locale.h>
#include <iostream>

int main() {
    std::cout << "Starting City Scape Simulation...\n";

    // Set locale for UTF-8 support
    setlocale(LC_ALL, "");

    // Create and run simulation
    CityScapeSimulation sim;
    return sim.run();
}
