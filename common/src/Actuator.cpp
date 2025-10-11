#include "../include/Actuator.h"
#include "../include/TerminalMatrix.h"
#include "../include/Tile.h"

Actuator::Actuator(int posX, int posY, const std::string& c, ActuatorType t, bool block, int color)
    : displayChar(c), x(posX), y(posY), type(t), blocking(block), colorPair(color) {
}

void Actuator::move(int dx, int dy, TerminalMatrix& matrix) {
    int newX = x + dx;
    int newY = y + dy;

    // Check if new position is valid
    Tile* newTile = matrix.getTile(newX, newY);
    if (!newTile || !newTile->isWalkable()) {
        return;  // Can't move there
    }

    // If new tile has an actuator, can't move there
    if (newTile->hasActuator()) {
        return;
    }

    // Clear current tile
    Tile* currentTile = matrix.getTile(x, y);
    if (currentTile) {
        currentTile->setActuator(nullptr);
    }

    // Move to new position
    x = newX;
    y = newY;
    newTile->setActuator(this);
}
