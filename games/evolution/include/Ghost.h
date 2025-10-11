#ifndef GHOST_H
#define GHOST_H

#include <string>
#include "TerminalMatrix.h"
#include "Actuator.h"

class Ghost : public Actuator {
public:
    // Constructor
    Ghost(int posX = 0, int posY = 0, const char* c = "👻");

    // Actions
    void move(int dx, int dy, TerminalMatrix& matrix);
    void killNearbyMice(TerminalMatrix& matrix);  // Kill mice on touch
};

#endif
