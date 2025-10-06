#include "Cat.h"
#include "Rodent.h"
#include "Actuator.h"
#include <cmath>
#include <random>

Cat::Cat(int posX, int posY, const std::string& c)
    : Actuator(posX, posY, c, ActuatorType::CAT), age(0), rodentsEaten(0),
      eatCooldown(0), moveCooldown(0), patrolDirection(0) {

    // Random starting patrol direction
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dir_dist(0, 3);
    patrolDirection = dir_dist(gen);
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> new_dir(0, 3);
        patrolDirection = new_dir(gen);
        return false;  // Movement failed
    }

    // Check if new position is valid (cats can walk over rocks and dead trees)
    Tile* tile = matrix.getTile(newX, newY);
    if (tile) {
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

        // Tile is empty - move into it
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
        // Invalid tile - change patrol direction
        std::random_device rd;
        std::mt19937 gen(rd());
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

void Cat::update(TerminalMatrix& matrix) {
    age++;

    // Decrease cooldowns
    if (eatCooldown > 0) eatCooldown--;
    if (moveCooldown > 0) moveCooldown--;

    // Try to eat nearby rodent first
    if (tryEatRodent(matrix)) {
        // Successfully ate - don't move this turn
        return;
    }

    // Cats move slower than rodents (every 3 ticks)
    if (moveCooldown > 0) {
        return;  // Still on cooldown, skip movement
    }

    // AI: Look for rodents, chase if found, else patrol
    int dx = 0, dy = 0;

    if (findClosestRodent(matrix, dx, dy)) {
        // Found a rodent - chase it!
        // Movement already set by findClosestRodent
    } else {
        // No rodent nearby - patrol pattern
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> change_dir(0, 9);

        // 10% chance to change patrol direction
        if (change_dir(gen) == 0) {
            std::uniform_int_distribution<> new_dir(0, 3);
            patrolDirection = new_dir(gen);
        }

        // Move in patrol direction: 0=N, 1=E, 2=S, 3=W
        switch(patrolDirection) {
            case 0: dy = -1; break;  // North
            case 1: dx = 1; break;   // East
            case 2: dy = 1; break;   // South
            case 3: dx = -1; break;  // West
        }
    }

    // Try to move
    if (dx != 0 || dy != 0) {
        bool moved = move(dx, dy, matrix);
        if (moved) {
            moveCooldown = 3;  // Wait 3 ticks before next move (only if moved)
        }
        // If movement failed, no cooldown - try again next tick with new direction
    }
}
