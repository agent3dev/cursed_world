#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>
#include <memory>

// Forward declarations
class TerminalMatrix;

// Base class for all simulations
// Provides common game loop, ncurses management, and input handling
class Simulation {
protected:
    std::string name;
    bool isRunning;
    bool paused;
    int tickCount;
    int frameDelayMs;  // Milliseconds between frames
    std::unique_ptr<TerminalMatrix> matrix;

    // Pure virtual methods that each simulation MUST implement
    virtual void initializeTerrain() = 0;      // Setup the world
    virtual void updateEntities() = 0;         // Update all game entities
    virtual void renderStats() = 0;            // Update dashboard/stats
    virtual void handleGameInput(int ch) {}    // Optional: game-specific input

    // Virtual methods with default implementations
    virtual void onPause() {}     // Called when game is paused
    virtual void onUnpause() {}   // Called when game is unpaused
    virtual void onQuit() {}      // Called before quitting

    // Common input handling (called before handleGameInput)
    bool handleCommonInput(int ch);

    // Helper methods
    void initializeNcurses();
    void cleanupNcurses();

public:
    Simulation(const std::string& simName, int frameDelay = 100);
    virtual ~Simulation();  // Defined in .cpp to allow incomplete type for unique_ptr

    // Main entry point - implements template method pattern
    virtual int run();

    // Template method hooks (can be overridden)
    virtual void initialize();
    virtual void cleanup();

    // Getters
    std::string getName() const { return name; }
    bool getIsRunning() const { return isRunning; }
    bool isPaused() const { return paused; }
    int getTickCount() const { return tickCount; }
    void setIsRunning(bool running) { isRunning = running; }
    void setFrameDelay(int ms) { frameDelayMs = ms; }
};

#endif
