#ifndef CITY_SCAPE_SIMULATION_H
#define CITY_SCAPE_SIMULATION_H

#include "../../../common/include/Simulation.h"
#include <memory>

// Forward declarations
class TerminalMatrix;
class Citizen;
class Vehicle;

// City Scape simulation: Urban navigation and traffic patterns
class CityScapeSimulation : public Simulation {
private:
    std::unique_ptr<TerminalMatrix> matrix;

    // Simulation state
    int tickCount;
    int citizenCount;
    int vehicleCount;
    bool paused;

    // Helper methods
    void initializeTerrain();
    void updateEntities();
    void handleInput(int ch);
    void renderStats();

public:
    CityScapeSimulation();
    ~CityScapeSimulation() override;

    // Simulation interface
    int run() override;
    void initialize() override;
    void cleanup() override;
};

#endif
