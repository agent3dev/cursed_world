#include "Cat.h"
#include "Rodent.h"
#include "Actuator.h"
#include <cmath>
#include <random>
#include <iostream>

int Cat::nextId = 0;

Cat::Cat(int posX, int posY, const std::string& c, const std::vector<double>& weights)
    : Actuator(posX, posY, c, ActuatorType::CAT), age(0), rodentsEaten(0),
      eatCooldown(0), moveCooldown(0), patrolDirection(0), alive(true),
      brain({10, 16, 9}), id(nextId++) {  // 10 inputs, 16 hidden, 9 outputs

    std::cout << "[DEBUG Cat] Creating cat " << id << " at (" << posX << ", " << posY << ")\n";

    // Random starting patrol direction
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dir_dist(0, 3);
    patrolDirection = dir_dist(gen);

    // Initialize brain with provided weights or random
    if (!weights.empty()) {
        brain.setWeights(weights);
        std::cout << "[DEBUG Cat] Cat " << id << " initialized with " << weights.size() << " weights\n";
    } else {
        brain.randomize(-1.0, 1.0);
        std::cout << "[DEBUG Cat] Cat " << id << " initialized with random weights\n";
    }
}

bool Cat::findClosestRodent(TerminalMatrix& matrix, int& outDx, int& outDy) {
    int closestDist = 999999;
    int targetDx = 0;
    int targetDy = 0;
    bool found = false;

    // Search in a 7x7 area around the cat
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            if (dx == 0 && dy == 0) continue;

            Tile* tile = matrix.getTile(getX() + dx, getY() + dy);
            if (tile && tile->hasActuator()) {
                Actuator* act = tile->getActuator();
                if (act && act->getType() == ActuatorType::RODENT) {
                    Rodent* rodent = static_cast<Rodent*>(act);
                    if (rodent->isAlive()) {
                        int dist = abs(dx) + abs(dy);  // Manhattan distance
                        if (dist < closestDist) {
                            closestDist = dist;
                            targetDx = dx;
                            targetDy = dy;
                            found = true;
                        }
                    }
                }
            }
        }
    }

    if (found) {
        // Return direction to move (one step towards target)
        outDx = (targetDx > 0) ? 1 : (targetDx < 0) ? -1 : 0;
        outDy = (targetDy > 0) ? 1 : (targetDy < 0) ? -1 : 0;
    }

    return found;
}

bool Cat::move(int dx, int dy, TerminalMatrix& matrix) {
    int newX = getX() + dx;
    int newY = getY() + dy;

    // Don't allow movement beyond screen bounds (no side walls, but also no offscreen)
    if (newX < 0 || newX >= matrix.getWidth() || newY < 0 || newY >= matrix.getHeight()) {
        // Hit edge - change patrol direction
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> new_dir(0, 3);
        patrolDirection = new_dir(gen);
        return false;  // Movement failed
    }

    // Check if new position is valid and walkable
    Tile* tile = matrix.getTile(newX, newY);
    if (tile && tile->isWalkable()) {
        // Check if there's a rodent on the target tile - if so, eat it!
        if (tile->hasActuator()) {
            Actuator* act = tile->getActuator();
            if (act && act->getType() == ActuatorType::RODENT) {
                Rodent* rodent = static_cast<Rodent*>(act);
                if (rodent->isAlive() && eatCooldown == 0) {
                    // Kill the rodent and move into its tile
                    rodent->kill();
                    tile->setActuator(nullptr);
                    rodentsEaten++;
                    eatCooldown = 30;  // Cooldown after eating
                }
            }
            // Don't move into tiles with other actuators (ghost, other cats)
            return false;
        }

        // Tile is empty and walkable - move into it
        // Remove actuator from old tile
        Tile* oldTile = matrix.getTile(getX(), getY());
        if (oldTile) {
            oldTile->setActuator(nullptr);
        }
        // Set new position
        setPosition(newX, newY);
        // Set actuator on new tile
        tile->setActuator(this);
        return true;  // Movement succeeded
    } else {
        // Invalid tile or not walkable (wall/obstacle) - change patrol direction
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> new_dir(0, 3);
        patrolDirection = new_dir(gen);
        return false;  // Movement failed
    }
}

bool Cat::tryEatRodent(TerminalMatrix& matrix) {
    // Can only eat if cooldown is over
    if (eatCooldown > 0) {
        return false;
    }

    // Check adjacent tiles for rodents (not current tile - cat can't eat itself!)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // Skip cat's own tile

            Tile* tile = matrix.getTile(getX() + dx, getY() + dy);

            if (tile && tile->hasActuator()) {
                Actuator* act = tile->getActuator();
                if (act && act->getType() == ActuatorType::RODENT) {
                    Rodent* rodent = static_cast<Rodent*>(act);
                    if (rodent->isAlive()) {
                        // Kill the rodent (no marker - just disappears)
                        tile->setActuator(nullptr);
                        rodentsEaten++;
                        eatCooldown = 30;  // Must wait 30 ticks before eating again
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

std::vector<double> Cat::getSurroundingInfo(TerminalMatrix& matrix) {
    std::vector<double> input;
    input.reserve(10);  // 8 surrounding + rodentsEaten + eatCooldown

    // Encode 8 surrounding tiles (NW, N, NE, W, E, SW, S, SE)
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int i = 0; i < 8; i++) {
        int checkX = getX() + dx[i];
        int checkY = getY() + dy[i];
        Tile* tile = matrix.getTile(checkX, checkY);

        double value = 0.0;  // Empty by default

        if (!tile) {
            // Out of bounds
            value = 7.0;
        } else if (tile->hasActuator()) {
            Actuator* act = tile->getActuator();
            if (act && act->getType() == ActuatorType::RODENT) {
                Rodent* rodent = static_cast<Rodent*>(act);
                if (rodent->isAlive()) {
                    value = 6.0;  // Rodent (prey!)
                }
            } else if (act && act->getType() == ActuatorType::CAT) {
                value = 8.0;  // Another cat (avoid)
            }
        } else {
            // Check terrain type
            TerrainType terrain = tile->getTerrainType();
            switch (terrain) {
                case TerrainType::EMPTY: value = 0.0; break;
                case TerrainType::PLANTS: value = 1.0; break;
                case TerrainType::SEEDLINGS: value = 2.0; break;
                case TerrainType::DEAD_TREES: value = 3.0; break;
                case TerrainType::ROCKS: value = 4.0; break;
                case TerrainType::SEED: value = 5.0; break;
                default: value = 0.0; break;
            }
        }

        input.push_back(value / 8.0);  // Normalize to 0-1 range
    }

    // Add rodents eaten (normalized to 0-1, assuming max ~20 per generation)
    input.push_back(std::min(rodentsEaten / 20.0, 1.0));

    // Add eat cooldown (normalized to 0-1, max 30 ticks)
    input.push_back(eatCooldown / 30.0);

    return input;
}

void Cat::update(TerminalMatrix& matrix) {
    if (!alive) return;

    age++;

    // Decrease cooldowns
    if (eatCooldown > 0) eatCooldown--;
    if (moveCooldown > 0) moveCooldown--;

    // Try to eat nearby rodent first
    if (tryEatRodent(matrix)) {
        // Successfully ate - don't move this turn
        return;
    }

    // Cats move at same speed as mice
    if (moveCooldown > 0) {
        return;  // Still on cooldown, skip movement
    }

    // Use neural network to decide movement
    std::vector<double> input = getSurroundingInfo(matrix);
    std::vector<double> output = brain.forward(input);

    // Find the best action (argmax)
    int bestAction = 0;
    double bestValue = output[0];
    for (int i = 1; i < output.size(); i++) {
        if (output[i] > bestValue) {
            bestValue = output[i];
            bestAction = i;
        }
    }

    // Map action to movement (8 directions + stay)
    // 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE, 8=Stay
    int dx = 0, dy = 0;
    switch(bestAction) {
        case 0: dx = -1; dy = -1; break;  // NW
        case 1: dx = 0; dy = -1; break;   // N
        case 2: dx = 1; dy = -1; break;   // NE
        case 3: dx = -1; dy = 0; break;   // W
        case 4: dx = 1; dy = 0; break;    // E
        case 5: dx = -1; dy = 1; break;   // SW
        case 6: dx = 0; dy = 1; break;    // S
        case 7: dx = 1; dy = 1; break;    // SE
        case 8: dx = 0; dy = 0; break;    // Stay
    }

    // Try to move
    if (dx != 0 || dy != 0) {
        bool moved = move(dx, dy, matrix);
        if (moved) {
            moveCooldown = 0;  // Same speed as mice (no cooldown)
        }
    }
}

Cat* Cat::reproduce(int posX, int posY) {
    // Create offspring with mutated brain
    std::vector<double> childWeights = brain.getWeights();

    // Mutate 5% of weights with ±0.3 adjustment
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> mutation_chance(0.0, 1.0);
    std::uniform_real_distribution<> mutation_amount(-0.3, 0.3);

    for (double& weight : childWeights) {
        if (mutation_chance(gen) < 0.05) {  // 5% mutation rate
            weight += mutation_amount(gen);
        }
    }

    Cat* child = new Cat(posX, posY, "🐈", childWeights);
    return child;
}
