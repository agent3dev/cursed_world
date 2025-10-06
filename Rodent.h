#ifndef RODENT_H
#define RODENT_H

#include <string>
#include <vector>
#include "TerminalMatrix.h"
#include "NeuralNetwork.h"
#include "Actuator.h"

class Rodent : public Actuator {
private:
    int x;
    int y;
    std::string displayChar;
    double energy;  // Energy level (dies at 0, can reproduce at high levels)
    int foodEaten;  // Fitness metric
    int age;  // Age in ticks
    bool alive;  // Is the rodent alive?
    NeuralNetwork brain;

    // Encode surrounding tiles into neural network input
    std::vector<double> getSurroundingInfo(TerminalMatrix& matrix);

public:
    // Constructor with optional neural network weights
    Rodent(int posX = 0, int posY = 0, const std::string& c = "🐀", const std::vector<double>& weights = {});

    // Getters
    int getX() const { return Actuator::getX(); }
    int getY() const { return Actuator::getY(); }
    const std::string& getChar() const { return Actuator::getChar(); }
    double getEnergy() const { return energy; }
    int getFoodEaten() const { return foodEaten; }
    int getAge() const { return age; }
    bool isAlive() const { return alive; }
    double getFitness() const { return foodEaten * 10.0 + age * 0.1; }  // Fitness = food priority + survival bonus
    NeuralNetwork& getBrain() { return brain; }
    const NeuralNetwork& getBrain() const { return brain; }

    // Actions
    void move(int dx, int dy, TerminalMatrix& matrix);
    void eat(TerminalMatrix& matrix);
    void update(TerminalMatrix& matrix);  // Main update logic (now uses neural network)
    bool canReproduce() const { return alive && energy >= 120.0; }  // Can reproduce at high energy
    Rodent* reproduce(int posX, int posY);  // Create offspring with mutated brain
    void kill() { alive = false; energy = 0.0; }  // Instantly kill the rodent

    // Setters
    void setPosition(int posX, int posY) { Actuator::setPosition(posX, posY); }
    void setChar(const std::string& c) { Actuator::setChar(c); }
};

#endif
