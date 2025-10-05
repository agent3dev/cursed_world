#include "Rodent.h"
#include <random>

Rodent::Rodent(int posX, int posY, const std::string& c)
    : x(posX), y(posY), displayChar(c), hunger(0) {
}

void Rodent::move(int dx, int dy, TerminalMatrix& matrix) {
    int newX = x + dx;
    int newY = y + dy;

    // Check if new position is valid and walkable
    Tile* tile = matrix.getTile(newX, newY);
    if (tile && tile->isWalkable()) {
        x = newX;
        y = newY;
    }
}

void Rodent::eat(TerminalMatrix& matrix) {
    Tile* tile = matrix.getTile(x, y);
    if (tile && tile->isEdible()) {
        // Eat the food
        tile->setEdible(false);
        tile->setChar(" ");
        tile->setTerrainType(TerrainType::EMPTY);
        hunger = 0;  // Reset hunger
    }
}

void Rodent::update(TerminalMatrix& matrix) {
    // Increase hunger over time
    hunger++;

    // Look for nearby food
    bool foundFood = false;
    int foodX = 0, foodY = 0;

    // Check surrounding tiles for food
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            Tile* tile = matrix.getTile(x + dx, y + dy);
            if (tile && tile->isEdible()) {
                foodX = dx;
                foodY = dy;
                foundFood = true;
                break;
            }
        }
        if (foundFood) break;
    }

    if (foundFood) {
        // Move toward food
        move(foodX != 0 ? (foodX > 0 ? 1 : -1) : 0,
             foodY != 0 ? (foodY > 0 ? 1 : -1) : 0,
             matrix);
    } else {
        // Random movement
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dir(-1, 1);

        move(dir(gen), dir(gen), matrix);
    }

    // Try to eat if on food
    eat(matrix);
}
