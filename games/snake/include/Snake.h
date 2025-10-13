#ifndef SNAKE_H
#define SNAKE_H

#include <vector>
#include <utility>

// Forward declaration
class TerminalMatrix;

enum class Direction {
    UP,
    DOWN,
    LEFT,
    RIGHT
};

struct Position {
    int x;
    int y;

    Position(int _x, int _y) : x(_x), y(_y) {}

    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

class Snake {
private:
    std::vector<Position> segments;  // segments[0] is head
    Direction direction;
    Direction nextDirection;  // Buffer next direction to prevent reversing
    bool growPending;
    bool alive;

public:
    Snake(int startX, int startY, Direction startDir);

    // Movement
    void setDirection(Direction dir);
    void move(TerminalMatrix& matrix);
    void grow() { growPending = true; }

    // State
    bool isAlive() const { return alive; }
    void kill() { alive = false; }
    int getLength() const { return segments.size(); }
    Position getHeadPosition() const { return segments[0]; }

    // Collision detection
    bool checkSelfCollision() const;
    bool occupiesPosition(int x, int y) const;

    // Rendering
    void render(TerminalMatrix& matrix);
    void clearFromMatrix(TerminalMatrix& matrix);
};

#endif
