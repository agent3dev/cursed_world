#include "../include/EvolutionSimulation.h"
#include <locale.h>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Starting Evolution Simulation...\n";

    // Set locale for UTF-8 support
    setlocale(LC_ALL, "");

    // Create and run simulation with command-line arguments
    EvolutionSimulation sim(argc, argv);
    return sim.run();
}
