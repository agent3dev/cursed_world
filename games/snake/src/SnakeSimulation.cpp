#include "../include/SnakeSimulation.h"
#include "../../../common/include/TerminalMatrix.h"
#include "../../../common/include/Tile.h"
#include <ncurses.h>
#include <random>
#include <sstream>
#include <iostream>

SnakeSimulation::SnakeSimulation()
    : Simulation("Snake Game - Classic Gameplay", 120),  // 120ms frame delay
      foodPosition(0, 0), score(0), gameOver(false),
      moveCounter(0), movesPerUpdate(2) {
}

SnakeSimulation::~SnakeSimulation() {
    // Destructor defined here where Snake is complete
}

void SnakeSimulation::initialize() {
    // Call base class initialization (sets up matrix, calls initializeTerrain)
    Simulation::initialize();

    // Create snake in the center
    int centerX = matrix->getWidth() / 2;
    int centerY = matrix->getHeight() / 2;
    snake = std::make_unique<Snake>(centerX, centerY, Direction::RIGHT);

    // Spawn initial food
    spawnFood();

    // Render snake and food
    snake->render(*matrix);
}

void SnakeSimulation::initializeTerrain() {
    if (!matrix) return;

    // Fill with empty walkable tiles
    for (int y = 1; y < matrix->getHeight() - 1; y++) {
        for (int x = 0; x < matrix->getWidth(); x++) {
            Tile* tile = matrix->getTile(x, y);
            if (tile) {
                tile->setChar("  ");
                tile->setWalkable(true);
            }
        }
    }

    // Add some obstacles for variety
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> obstacle_dist(0, 100);

    int numObstacles = (matrix->getWidth() * matrix->getHeight()) / 50;  // ~2% obstacles
    int placed = 0;

    while (placed < numObstacles) {
        std::uniform_int_distribution<> x_dist(2, matrix->getWidth() - 3);
        std::uniform_int_distribution<> y_dist(2, matrix->getHeight() - 3);

        int x = x_dist(gen);
        int y = y_dist(gen);

        // Don't place obstacles near center where snake starts
        int centerX = matrix->getWidth() / 2;
        int centerY = matrix->getHeight() / 2;
        if (abs(x - centerX) < 5 && abs(y - centerY) < 5) {
            continue;
        }

        Tile* tile = matrix->getTile(x, y);
        if (tile) {
            tile->setChar("🪨");
            tile->setWalkable(false);
            placed++;
        }
    }

    // Draw border
    matrix->margin("🧱");
}

void SnakeSimulation::spawnFood() {
    if (!matrix) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x_dist(1, matrix->getWidth() - 2);
    std::uniform_int_distribution<> y_dist(1, matrix->getHeight() - 2);

    // Find a valid empty position
    int attempts = 0;
    while (attempts < 100) {
        int x = x_dist(gen);
        int y = y_dist(gen);

        Tile* tile = matrix->getTile(x, y);
        if (tile && tile->isWalkable() && !snake->occupiesPosition(x, y)) {
            foodPosition = Position(x, y);
            tile->setChar("🍎");
            return;
        }
        attempts++;
    }

    // Fallback: find any valid position
    for (int y = 1; y < matrix->getHeight() - 1; y++) {
        for (int x = 0; x < matrix->getWidth(); x++) {
            Tile* tile = matrix->getTile(x, y);
            if (tile && tile->isWalkable() && !snake->occupiesPosition(x, y)) {
                foodPosition = Position(x, y);
                tile->setChar("🍎");
                return;
            }
        }
    }
}

bool SnakeSimulation::isFoodPosition(int x, int y) const {
    return foodPosition.x == x && foodPosition.y == y;
}

void SnakeSimulation::updateEntities() {
    if (!matrix || !snake || gameOver) return;

    // Control snake movement speed (move every N ticks)
    moveCounter++;
    if (moveCounter < movesPerUpdate) {
        return;
    }
    moveCounter = 0;

    // Move snake
    snake->move(*matrix);

    // Check if snake died
    if (!snake->isAlive()) {
        gameOver = true;
        snake->render(*matrix);
        return;
    }

    // Check if snake ate food
    Position newHead = snake->getHeadPosition();
    if (newHead == foodPosition) {
        score += 10;
        snake->grow();
        spawnFood();

        // Speed up slightly as score increases
        if (score % 50 == 0 && movesPerUpdate > 1) {
            movesPerUpdate--;
        }
    }

    // Render snake
    snake->render(*matrix);
}

void SnakeSimulation::renderStats() {
    if (!matrix) return;

    std::stringstream ss;
    ss << "Score: " << score
       << " | Length: " << (snake ? snake->getLength() : 0)
       << " | Speed: " << (3 - movesPerUpdate)
       << " | " << (paused ? "[PAUSED]" : (gameOver ? "[GAME OVER - Press Q to quit]" : "[PLAYING]"));

    matrix->setDashboard(ss.str());
}

void SnakeSimulation::handleGameInput(int ch) {
    if (!snake || gameOver) return;

    // Handle arrow keys for snake direction
    switch (ch) {
        case KEY_UP:
            snake->setDirection(Direction::UP);
            break;
        case KEY_DOWN:
            snake->setDirection(Direction::DOWN);
            break;
        case KEY_LEFT:
            snake->setDirection(Direction::LEFT);
            break;
        case KEY_RIGHT:
            snake->setDirection(Direction::RIGHT);
            break;
        // Also support WASD
        case 'w':
        case 'W':
            snake->setDirection(Direction::UP);
            break;
        case 's':
        case 'S':
            snake->setDirection(Direction::DOWN);
            break;
        case 'a':
        case 'A':
            snake->setDirection(Direction::LEFT);
            break;
        case 'd':
        case 'D':
            snake->setDirection(Direction::RIGHT);
            break;
    }
}

void SnakeSimulation::onPause() {
    // Could add pause-specific behavior here
}

void SnakeSimulation::onUnpause() {
    // Could add unpause-specific behavior here
}
