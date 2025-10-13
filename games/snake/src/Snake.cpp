#include "../include/Snake.h"
#include "../../../common/include/TerminalMatrix.h"
#include "../../../common/include/Tile.h"

Snake::Snake(int startX, int startY, Direction startDir)
    : direction(startDir), nextDirection(startDir), growPending(false), alive(true) {
    // Start with 3 segments
    segments.push_back(Position(startX, startY));

    // Add tail segments based on starting direction
    switch (startDir) {
        case Direction::RIGHT:
            segments.push_back(Position(startX - 1, startY));
            segments.push_back(Position(startX - 2, startY));
            break;
        case Direction::LEFT:
            segments.push_back(Position(startX + 1, startY));
            segments.push_back(Position(startX + 2, startY));
            break;
        case Direction::DOWN:
            segments.push_back(Position(startX, startY - 1));
            segments.push_back(Position(startX, startY - 2));
            break;
        case Direction::UP:
            segments.push_back(Position(startX, startY + 1));
            segments.push_back(Position(startX, startY + 2));
            break;
    }
}

void Snake::setDirection(Direction dir) {
    // Prevent reversing into self
    bool isOpposite = false;
    if (direction == Direction::UP && dir == Direction::DOWN) isOpposite = true;
    if (direction == Direction::DOWN && dir == Direction::UP) isOpposite = true;
    if (direction == Direction::LEFT && dir == Direction::RIGHT) isOpposite = true;
    if (direction == Direction::RIGHT && dir == Direction::LEFT) isOpposite = true;

    if (!isOpposite) {
        nextDirection = dir;
    }
}

void Snake::move(TerminalMatrix& matrix) {
    if (!alive) return;

    // Update direction from buffered input
    direction = nextDirection;

    // Calculate new head position
    Position oldHead = segments[0];
    Position newHead = oldHead;

    switch (direction) {
        case Direction::UP:
            newHead.y--;
            break;
        case Direction::DOWN:
            newHead.y++;
            break;
        case Direction::LEFT:
            newHead.x--;
            break;
        case Direction::RIGHT:
            newHead.x++;
            break;
    }

    // Check wall collision
    Tile* newTile = matrix.getTile(newHead.x, newHead.y);
    if (!newTile || !newTile->isWalkable()) {
        alive = false;
        return;
    }

    // Add new head
    segments.insert(segments.begin(), newHead);

    // Remove tail if not growing
    if (growPending) {
        growPending = false;
    } else {
        // Clear old tail from matrix
        Position tail = segments.back();
        Tile* tailTile = matrix.getTile(tail.x, tail.y);
        if (tailTile) {
            tailTile->setChar("  ");
        }
        segments.pop_back();
    }

    // Check self collision (after moving)
    if (checkSelfCollision()) {
        alive = false;
    }
}

bool Snake::checkSelfCollision() const {
    Position head = segments[0];
    for (size_t i = 1; i < segments.size(); i++) {
        if (segments[i] == head) {
            return true;
        }
    }
    return false;
}

bool Snake::occupiesPosition(int x, int y) const {
    for (const auto& segment : segments) {
        if (segment.x == x && segment.y == y) {
            return true;
        }
    }
    return false;
}

void Snake::render(TerminalMatrix& matrix) {
    if (segments.empty()) return;

    // Render head
    Tile* headTile = matrix.getTile(segments[0].x, segments[0].y);
    if (headTile) {
        headTile->setChar(alive ? "🐍" : "💀");
    }

    // Render body
    for (size_t i = 1; i < segments.size(); i++) {
        Tile* tile = matrix.getTile(segments[i].x, segments[i].y);
        if (tile) {
            tile->setChar("🟩");
        }
    }
}

void Snake::clearFromMatrix(TerminalMatrix& matrix) {
    for (const auto& segment : segments) {
        Tile* tile = matrix.getTile(segment.x, segment.y);
        if (tile) {
            tile->setChar("  ");
        }
    }
}
