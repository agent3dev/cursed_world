#include "../include/Citizen.h"
#include "../../../common/include/TerminalMatrix.h"
#include "../../../common/include/Tile.h"
#include <cmath>
#include <algorithm>

Citizen::Citizen(int startX, int startY, const char* emoji)
    : Actuator(startX, startY, emoji, ActuatorType::CITIZEN), destX(startX), destY(startY),
      patience(1000), reachedDest(false) {
}

Citizen::~Citizen() {
}

void Citizen::calculateNextMove(TerminalMatrix& matrix) {
    if (reachedDest || patience <= 0) {
        return;
    }

    // Simple pathfinding: move towards destination
    int dx = destX - x;
    int dy = destY - y;

    int moveX = 0;
    int moveY = 0;

    // Prioritize larger delta
    if (std::abs(dx) > std::abs(dy)) {
        moveX = (dx > 0) ? 1 : -1;
    } else if (std::abs(dy) > 0) {
        moveY = (dy > 0) ? 1 : -1;
    }

    int newX = x + moveX;
    int newY = y + moveY;

    // Try to move
    if (isValidMove(newX, newY, matrix)) {
        move(moveX, moveY, matrix);
        patience--;
    } else {
        // Try alternate route
        if (moveX != 0 && isValidMove(x, y + ((dy > 0) ? 1 : -1), matrix)) {
            move(0, (dy > 0) ? 1 : -1, matrix);
            patience--;
        } else if (moveY != 0 && isValidMove(x + ((dx > 0) ? 1 : -1), y, matrix)) {
            move((dx > 0) ? 1 : -1, 0, matrix);
            patience--;
        } else {
            patience -= 2;  // Penalize being stuck
        }
    }

    // Check if reached destination
    if (x == destX && y == destY) {
        reachedDest = true;
    }
}

bool Citizen::isValidMove(int newX, int newY, TerminalMatrix& matrix) {
    Tile* tile = matrix.getTile(newX, newY);
    if (!tile) return false;
    if (!tile->isWalkable()) return false;
    if (tile->hasActuator()) return false;  // Don't walk into other entities
    return true;
}

void Citizen::update(TerminalMatrix& matrix) {
    calculateNextMove(matrix);
}

void Citizen::setDestination(int x, int y) {
    destX = x;
    destY = y;
    reachedDest = false;
    patience = 1000;
}
