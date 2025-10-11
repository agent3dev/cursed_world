#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>

// Base class for all simulations
// Provides a common interface for different simulation types
class Simulation {
protected:
    std::string name;
    bool isRunning;

public:
    Simulation(const std::string& simName) : name(simName), isRunning(false) {}
    virtual ~Simulation() = default;

    // Pure virtual methods that each simulation must implement
    virtual int run() = 0;  // Main simulation loop, returns exit code
    virtual void initialize() = 0;  // Setup the simulation
    virtual void cleanup() = 0;  // Cleanup resources

    // Common methods
    std::string getName() const { return name; }
    bool getIsRunning() const { return isRunning; }
    void setIsRunning(bool running) { isRunning = running; }
};

#endif
