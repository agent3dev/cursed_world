#ifndef CITY_SCAPE_SIMULATION_H
#define CITY_SCAPE_SIMULATION_H

#include "../../../common/include/Simulation.h"

// City Scape simulation: Urban navigation and traffic patterns
class CityScapeSimulation : public Simulation {
private:
    // Game-specific state
    int citizenCount;
    int vehicleCount;

protected:
    // Implement pure virtual methods from Simulation base class
    void initializeTerrain() override;
    void updateEntities() override;
    void renderStats() override;
    void handleGameInput(int ch) override;

public:
    CityScapeSimulation();
    ~CityScapeSimulation() override = default;

    // Override initialize to add game-specific setup
    void initialize() override;
};

#endif
