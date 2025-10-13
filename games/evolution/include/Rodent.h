#ifndef RODENT_H
#define RODENT_H

#include <string>
#include <vector>
#include "../../common/include/TerminalMatrix.h"
#include "NeuralNetwork.h"
#include "../../common/include/Actuator.h"

// Helper struct for distance-based sensing ("smell")
struct NearestEntity {
    int dx, dy;        // Direction vector to target
    int distance;      // Manhattan distance
    bool found;        // Whether any entity was found

    NearestEntity() : dx(0), dy(0), distance(999999), found(false) {}
};

class Rodent : public Actuator {
private:
    double energy;  // Energy level (dies at 0, can reproduce at high levels)
    int foodEaten;  // Fitness metric
    int age;  // Age in ticks
    bool alive;  // Is the rodent alive?
    NeuralNetwork brain;
    int ticksSinceLastPoop;  // Track ticks since last poop
    int id;  // Unique ID for debugging
    static int nextId;  // Static counter for IDs

    // Encode surrounding tiles into neural network input
    std::vector<double> getSurroundingInfo(TerminalMatrix& matrix);

    // Distance-based sensing ("smell") - find nearest entities
    NearestEntity findNearestCat(TerminalMatrix& matrix, int search_radius = 20);
    NearestEntity findNearestPeer(TerminalMatrix& matrix, int search_radius = 15);
    NearestEntity findNearestFood(TerminalMatrix& matrix, int search_radius = 15);

public:
    // Constructor with optional neural network weights
    Rodent(int posX = 0, int posY = 0, const char* c = "🐀", const std::vector<double>& weights = {});

    // Getters
    double getEnergy() const { return energy; }
    int getFoodEaten() const { return foodEaten; }
    int getId() const { return id; }
    int getAge() const { return age; }
    bool isAlive() const { return alive; }
    double getFitness() const { return foodEaten * 10.0 + age * 0.1; }  // Fitness = food priority + survival bonus
    NeuralNetwork& getBrain() { return brain; }
    const NeuralNetwork& getBrain() const { return brain; }

    // Actions
    bool move(int dx, int dy, TerminalMatrix& matrix);
    void eat(TerminalMatrix& matrix);
    void poop(TerminalMatrix& matrix);  // Leave seed that will grow into seedling
    void update(TerminalMatrix& matrix);  // Main update logic (now uses neural network)
    bool canReproduce() const { return alive && energy >= 120.0; }  // Can reproduce at high energy
    Rodent* reproduce(int posX, int posY);  // Create offspring with mutated brain
    void kill() { alive = false; energy = 0.0; }  // Instantly kill the rodent
    void resetMemory() { brain.resetHiddenState(); }  // Reset recurrent memory

    // Setters
    void setPosition(int posX, int posY) { Actuator::setPosition(posX, posY); }
    void setChar(const char* c) { Actuator::setChar(c); }
};

#endif
