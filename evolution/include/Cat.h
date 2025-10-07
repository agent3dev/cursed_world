#ifndef CAT_H
#define CAT_H

#include <string>
#include <vector>
#include "TerminalMatrix.h"
#include "NeuralNetwork.h"
#include "Actuator.h"

class Cat : public Actuator {
private:
    int x;
    int y;
    std::string displayChar;
    int age;
    int rodentsEaten;
    int eatCooldown;  // Ticks before can eat again
    int moveCooldown;  // Ticks before can move again
    int patrolDirection;  // 0-3 for N/E/S/W patrol
    bool alive;  // Is the cat alive?
    NeuralNetwork brain;  // Neural network for decision making
    int id;  // Unique ID for debugging
    static int nextId;  // Static counter for IDs

    // Find closest rodent (for encoding into neural network input)
    bool findClosestRodent(TerminalMatrix& matrix, int& outDx, int& outDy);

    // Encode surrounding tiles into neural network input
    std::vector<double> getSurroundingInfo(TerminalMatrix& matrix);

public:
    // Constructor with optional neural network weights
    Cat(int posX = 0, int posY = 0, const std::string& c = "🐈", const std::vector<double>& weights = {});

    // Getters
    int getX() const { return Actuator::getX(); }
    int getY() const { return Actuator::getY(); }
    const std::string& getChar() const { return Actuator::getChar(); }
    int getAge() const { return age; }
    int getRodentsEaten() const { return rodentsEaten; }
    int getId() const { return id; }
    bool isAlive() const { return alive; }
    double getFitness() const { return rodentsEaten * 100.0 + age * 0.1; }  // Fitness based on kills + survival
    NeuralNetwork& getBrain() { return brain; }
    const NeuralNetwork& getBrain() const { return brain; }
    void resetRodentsEaten() { rodentsEaten = 0; }

    // Actions
    bool move(int dx, int dy, TerminalMatrix& matrix);  // Returns true if movement succeeded
    bool tryEatRodent(TerminalMatrix& matrix);  // Check if rodent nearby and eat it
    void update(TerminalMatrix& matrix);  // Main update logic (now uses neural network)
    Cat* reproduce(int posX, int posY);  // Create offspring with mutated brain
    void kill() { alive = false; }  // Instantly kill the cat
    void resetMemory() { brain.resetHiddenState(); }  // Reset recurrent memory

    // Setters
    void setPosition(int posX, int posY) { Actuator::setPosition(posX, posY); }
    void setChar(const std::string& c) { Actuator::setChar(c); }
};

#endif
