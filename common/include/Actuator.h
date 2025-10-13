#ifndef ACTUATOR_H
#define ACTUATOR_H

#include <string>

// Forward declaration
class TerminalMatrix;

enum class ActuatorType {
    CHARACTER,
    NPC,
    TRAP,
    ITEM,
    ENEMY,
    RODENT,
    CAT,
    CITIZEN,
    VEHICLE
};

class Actuator {
protected:  // Changed from private to protected so derived classes can access
    std::string displayChar;
    int x;
    int y;
    ActuatorType type;
    bool blocking;
    int colorPair;

public:
    Actuator(int posX = 0, int posY = 0, const std::string& c = "@", ActuatorType t = ActuatorType::CHARACTER, bool block = true, int color = 0);
    virtual ~Actuator() = default;  // Virtual destructor

    // Getters
    const std::string& getChar() const { return displayChar; }
    int getX() const { return x; }
    int getY() const { return y; }
    ActuatorType getType() const { return type; }
    bool isBlocking() const { return blocking; }
    int getColorPair() const { return colorPair; }

    // Setters
    void setChar(const std::string& c) { displayChar = c; }
    void setPosition(int posX, int posY) { x = posX; y = posY; }
    void setX(int posX) { x = posX; }
    void setY(int posY) { y = posY; }
    void setType(ActuatorType t) { type = t; }
    void setBlocking(bool block) { blocking = block; }
    void setColorPair(int color) { colorPair = color; }

    // Virtual methods for derived classes
    virtual void update(TerminalMatrix& matrix) {}  // Override in derived classes
    virtual bool move(int dx, int dy, TerminalMatrix& matrix);  // Default move implementation, returns success
};

#endif
