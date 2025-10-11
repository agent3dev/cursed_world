#ifndef CITIZEN_H
#define CITIZEN_H

#include "../../../common/include/Actuator.h"
#include <string>

// Forward declaration
class TerminalMatrix;

// Citizen: Pedestrian navigating the city
class Citizen : public Actuator {
private:
    int destX, destY;      // Destination coordinates
    int patience;          // How long until they give up
    bool reachedDest;      // Whether they reached their destination

    // Movement AI
    void calculateNextMove(TerminalMatrix& matrix);
    bool isValidMove(int newX, int newY, TerminalMatrix& matrix);

public:
    Citizen(int startX, int startY, const char* emoji);
    ~Citizen() override;

    void update(TerminalMatrix& matrix) override;
    void setDestination(int x, int y);
    bool hasReachedDestination() const { return reachedDest; }
    int getPatience() const { return patience; }
};

#endif
