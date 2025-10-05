#ifndef RODENT_H
#define RODENT_H

#include <string>
#include "TerminalMatrix.h"

class Rodent {
private:
    int x;
    int y;
    std::string displayChar;
    int hunger;  // 0 = full, higher = hungrier

public:
    Rodent(int posX = 0, int posY = 0, const std::string& c = "🐀");

    // Getters
    int getX() const { return x; }
    int getY() const { return y; }
    const std::string& getChar() const { return displayChar; }
    int getHunger() const { return hunger; }

    // Actions
    void move(int dx, int dy, TerminalMatrix& matrix);
    void eat(TerminalMatrix& matrix);
    void update(TerminalMatrix& matrix);  // Main update logic

    // Setters
    void setPosition(int posX, int posY) { x = posX; y = posY; }
    void setChar(const std::string& c) { displayChar = c; }
};

#endif
