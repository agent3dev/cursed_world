#ifndef SNAKE_SIMULATION_H
#define SNAKE_SIMULATION_H

#include "../../../common/include/Simulation.h"
#include "Snake.h"
#include <memory>

// Classic Snake game simulation
class SnakeSimulation : public Simulation {
private:
    std::unique_ptr<Snake> snake;
    Position foodPosition;
    int score;
    bool gameOver;
    int moveCounter;  // For controlling snake speed
    int movesPerUpdate;  // How many ticks between snake moves

    // Helper methods
    void spawnFood();
    bool isFoodPosition(int x, int y) const;

protected:
    // Implement pure virtual methods from Simulation base class
    void initializeTerrain() override;
    void updateEntities() override;
    void renderStats() override;
    void handleGameInput(int ch) override;

    // Override hooks
    void onPause() override;
    void onUnpause() override;

public:
    SnakeSimulation();
    ~SnakeSimulation() override;

    // Override initialize to add game-specific setup
    void initialize() override;
};

#endif
