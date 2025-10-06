#ifndef GHOST_H
#define GHOST_H

#include <string>
#include "TerminalMatrix.h"
#include "Actuator.h"

class Ghost : public Actuator {
private:
    int x;
    int y;
    std::string displayChar;

public:
    // Constructor
    Ghost(int posX = 0, int posY = 0, const std::string& c = "👻");

    // Getters
    int getX() const { return Actuator::getX(); }
    int getY() const { return Actuator::getY(); }
    const std::string& getChar() const { return Actuator::getChar(); }

    // Actions
    void move(int dx, int dy, TerminalMatrix& matrix);
    void killNearbyMice(TerminalMatrix& matrix);  // Kill mice on touch

    // Setters
    void setPosition(int posX, int posY) { Actuator::setPosition(posX, posY); }
    void setChar(const std::string& c) { Actuator::setChar(c); }
};

#endif
