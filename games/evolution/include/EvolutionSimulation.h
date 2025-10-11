#ifndef EVOLUTION_SIMULATION_H
#define EVOLUTION_SIMULATION_H

#include "Simulation.h"
#include <vector>

// Forward declaration to avoid circular dependencies
class TerminalMatrix;
class PopulationManager;
class Ghost;

// Evolution simulation: Mice vs Cats with neural networks
class EvolutionSimulation : public Simulation {
private:
    // Loaded brain weights
    std::vector<double> bestMouseWeights;
    std::vector<double> bestCatWeights;

    // Helper methods
    void loadBrains();
    void saveBrains();
    void printSummary();

public:
    EvolutionSimulation();
    ~EvolutionSimulation() override;

    // Simulation interface
    int run() override;
    void initialize() override;
    void cleanup() override;
};

#endif
