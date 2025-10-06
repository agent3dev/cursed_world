#include "Rodent.h"
#include "Actuator.h"
#include <cmath>

Rodent::Rodent(int posX, int posY, const std::string& c, const std::vector<double>& weights)
    : Actuator(posX, posY, c, ActuatorType::RODENT), energy(100.0), foodEaten(0), age(0), alive(true),
      brain({9, 16, 9}) {  // Input: 9 (8 tiles + energy), Hidden: 16, Output: 9 (8 directions + stay)

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

    // 8 surrounding tiles, each encoded as single terrain type value
    // Order: NW, N, NE, W, E, SW, S, SE
    const int dx_order[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy_order[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int i = 0; i < 8; i++) {
        int dx = dx_order[i];
        int dy = dy_order[i];

        Tile* tile = matrix.getTile(getX() + dx, getY() + dy);

        if (!tile) {
            // Out of bounds = 6
            info.push_back(6.0);
        } else {
            // Check for cat first (highest priority)
            if (tile->hasActuator()) {
                Actuator* act = tile->getActuator();
                if (act && act->getType() == ActuatorType::CAT) {
                    info.push_back(5.0);  // CAT = 5
                    continue;
                }
            }

            // Check if wall/obstacle
            if (!tile->isWalkable()) {
                info.push_back(6.0);  // WALL = 6
            } else {
                // Encode terrain type: 0=EMPTY, 1=PLANTS, 2=SEEDLINGS, 3=DEAD_TREES, 4=ROCKS
                TerrainType terrain = tile->getTerrainType();
                info.push_back(static_cast<double>(terrain));
            }
        }
    }

    // Add normalized energy level as 9th input (0-1 range, max energy = 150)
    info.push_back(energy / 150.0);

    return info;  // 9 inputs
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

    // Output interpretation: 9 outputs for 8 directions + stay
    // 0=NW, 1=N, 2=NE, 3=W, 4=E, 5=SW, 6=S, 7=SE, 8=STAY

    // Find the highest output (argmax)
    int bestAction = 0;
    double bestValue = output[0];
    for (int i = 1; i < 9; i++) {
        if (output[i] > bestValue) {
            bestValue = output[i];
            bestAction = i;
        }
    }

    // Convert action to movement
    const int dx_actions[] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};  // NW, N, NE, W, E, SW, S, SE, STAY
    const int dy_actions[] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};

    int dx = dx_actions[bestAction];
    int dy = dy_actions[bestAction];

    // Try to move (unless STAY)
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
