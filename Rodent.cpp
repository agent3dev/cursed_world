#include "Rodent.h"
#include "Actuator.h"
#include <cmath>

Rodent::Rodent(int posX, int posY, const std::string& c, const std::vector<double>& weights)
    : Actuator(posX, posY, c, ActuatorType::RODENT), energy(100.0), foodEaten(0), age(0), alive(true),
      brain({32, 16, 8, 3}) {  // Input: 32 (8 tiles * 4 features), Hidden: 16, 8, Output: 3 (dx, dy, eat)

    if (weights.empty()) {
        // No weights provided, randomize
        brain.randomize(-1.0, 1.0);
    } else {
        // Use provided weights (for evolution)
        brain.setWeights(weights);
    }
}

std::vector<double> Rodent::getSurroundingInfo(TerminalMatrix& matrix) {
    std::vector<double> info;

    // 8 surrounding tiles (4 features each) = 32 features
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // Skip center

            Tile* tile = matrix.getTile(getX() + dx, getY() + dy);

            if (tile) {
                info.push_back(tile->isWalkable() ? 1.0 : 0.0);
                info.push_back(tile->isEdible() ? 1.0 : 0.0);
                info.push_back(tile->isTransparent() ? 1.0 : 0.0);

                // NEW: Cat detection - is there a cat on this tile?
                bool hasCat = false;
                if (tile->hasActuator()) {
                    Actuator* act = tile->getActuator();
                    if (act && act->getType() == ActuatorType::CAT) {
                        hasCat = true;
                    }
                }
                info.push_back(hasCat ? 1.0 : 0.0);
            } else {
                info.push_back(0.0);
                info.push_back(0.0);
                info.push_back(0.0);
                info.push_back(0.0);
            }
        }
    }

    return info;  // 8 tiles * 4 features = 32 inputs
}

void Rodent::move(int dx, int dy, TerminalMatrix& matrix) {
    int newX = getX() + dx;
    int newY = getY() + dy;

    // Don't allow movement beyond screen bounds (no side walls, but also no offscreen)
    if (newX < 0 || newX >= matrix.getWidth() || newY < 0 || newY >= matrix.getHeight()) {
        return;  // Can't move offscreen
    }

    // Check if new position is valid and walkable
    Tile* tile = matrix.getTile(newX, newY);
    if (tile && tile->isWalkable()) {
        // Movement costs energy
        energy -= 0.2;

        // Remove actuator from old tile
        Tile* oldTile = matrix.getTile(getX(), getY());
        if (oldTile) {
            oldTile->setActuator(nullptr);
        }
        // Set new position
        setPosition(newX, newY);
        // Set actuator on new tile
        if (tile) {
            tile->setActuator(this);
        }
    }
}

void Rodent::eat(TerminalMatrix& matrix) {
    Tile* tile = matrix.getTile(getX(), getY());
    if (tile && tile->isEdible()) {
        // Eat the food
        tile->setEdible(false);
        tile->setChar("  ");  // Two spaces to match emoji width
        tile->setTerrainType(TerrainType::EMPTY);
        energy += 40.0;  // Gain energy from food
        if (energy > 150.0) energy = 150.0;  // Cap at 150
        foodEaten++;  // Increase fitness
    }
}

void Rodent::update(TerminalMatrix& matrix) {
    if (!alive) return;

    // Passive energy drain
    energy -= 0.05;
    age++;

    // Check for starvation
    if (energy <= 0.0) {
        alive = false;
        energy = 0.0;
        return;
    }

    // Get surrounding tile information
    std::vector<double> input = getSurroundingInfo(matrix);

    // Feed through neural network
    std::vector<double> output = brain.forward(input);

    // Output interpretation:
    // output[0] = dx movement (-1 to 1)
    // output[1] = dy movement (-1 to 1)
    // output[2] = eat action (-1 to 1, >0 means try to eat)

    // Convert continuous output to discrete movement
    int dx = 0, dy = 0;
    if (output[0] > 0.1) dx = 1;
    else if (output[0] < -0.1) dx = -1;

    if (output[1] > 0.1) dy = 1;
    else if (output[1] < -0.1) dy = -1;

    // Try to move
    if (dx != 0 || dy != 0) {
        move(dx, dy, matrix);
    }

    // Always try to eat if standing on food (automatic)
    eat(matrix);
}

Rodent* Rodent::reproduce(int posX, int posY) {
    if (!canReproduce()) return nullptr;

    // Cost energy to reproduce (expensive!)
    energy -= 60.0;

    // Create offspring with mutated brain
    std::vector<double> parentWeights = brain.getWeights();
    Rodent* offspring = new Rodent(posX, posY, "🐀", parentWeights);

    // Mutate the offspring's brain (5% mutation rate, small changes)
    offspring->brain.mutate(0.05, 0.3);

    return offspring;
}
