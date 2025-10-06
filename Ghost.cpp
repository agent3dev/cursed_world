#include "Ghost.h"
#include "Rodent.h"
#include "Actuator.h"

Ghost::Ghost(int posX, int posY, const std::string& c)
    : Actuator(posX, posY, c, ActuatorType::CHARACTER) {}

void Ghost::move(int dx, int dy, TerminalMatrix& matrix) {
    int newX = getX() + dx;
    int newY = getY() + dy;

    // Don't allow movement beyond screen bounds
    if (newX < 0 || newX >= matrix.getWidth() || newY < 0 || newY >= matrix.getHeight()) {
        return;  // Can't move offscreen
    }

    // Ghost can move through everything (like cats over obstacles)
    Tile* tile = matrix.getTile(newX, newY);
    if (tile) {
        // Remove ghost from old tile
        Tile* oldTile = matrix.getTile(getX(), getY());
        if (oldTile && oldTile->getActuator() == this) {
            oldTile->setActuator(nullptr);
        }

        // Set new position
        setPosition(newX, newY);

        // Only set actuator if tile is empty, otherwise ghost floats over
        if (!tile->hasActuator()) {
            tile->setActuator(this);
        } else {
            // If there's something here, we'll handle it in killNearbyMice
            // For now, just update position without setting actuator
            // The ghost will appear to float over the other entity
        }
    }
}

void Ghost::killNearbyMice(TerminalMatrix& matrix) {
    // Check current tile and adjacent tiles for mice
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            Tile* tile = matrix.getTile(getX() + dx, getY() + dy);

            if (tile && tile->hasActuator()) {
                Actuator* act = tile->getActuator();
                if (act && act->getType() == ActuatorType::RODENT) {
                    Rodent* rodent = static_cast<Rodent*>(act);
                    if (rodent->isAlive()) {
                        // Kill the rodent
                        rodent->kill();  // We'll need to add this method
                        tile->setActuator(nullptr);
                        tile->setChar("🪦");  // Tombstone
                    }
                }
            }
        }
    }
}
