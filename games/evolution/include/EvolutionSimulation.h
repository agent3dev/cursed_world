#ifndef EVOLUTION_SIMULATION_H
#define EVOLUTION_SIMULATION_H

#include "../../../common/include/Simulation.h"
#include "../../../common/include/ComputeConfig.h"
#include <vector>
#include <chrono>
#include <memory>

// Forward declarations
class PopulationManager;
class Ghost;

// Evolution simulation: Mice vs Cats with neural networks
class EvolutionSimulation : public Simulation {
private:
    // Brain weights from previous runs
    std::vector<double> bestMouseWeights;
    std::vector<double> bestCatWeights;

    // Compute configuration
    std::unique_ptr<ComputeConfig> computeConfig;

    // Game-specific state
    std::unique_ptr<PopulationManager> popManager;
    std::unique_ptr<Ghost> playerGhost;

    // Wall animation timer
    std::chrono::high_resolution_clock::time_point lastWallToggle;
    int wallAnimationInterval;

    // Helper methods
    void loadBrains();
    void saveBrains();

protected:
    // Implement pure virtual methods from Simulation base class
    void initializeTerrain() override;
    void updateEntities() override;
    void renderStats() override;
    void handleGameInput(int ch) override;

    // Override hooks
    void onQuit() override;

public:
    EvolutionSimulation(int argc = 0, char* argv[] = nullptr);
    ~EvolutionSimulation() override;

    // Override initialize to add game-specific setup
    void initialize() override;

    // Set compute configuration (can be called before initialize)
    void setComputeConfig(std::unique_ptr<ComputeConfig> config);
};

#endif
