#ifndef POPULATION_MANAGER_H
#define POPULATION_MANAGER_H

#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include "Rodent.h"
#include "Cat.h"
#include "TerminalMatrix.h"

class PopulationManager {
private:
    std::vector<std::unique_ptr<Rodent>> population;
    std::vector<std::unique_ptr<Cat>> cats;
    int generation;
    int maxPopulation;
    int generationLength;  // Ticks per generation
    int currentTick;
    int maxCats;
    int totalDeaths;  // Track cumulative deaths
    std::ofstream debugLog;  // Debug log for death tracking

public:
    PopulationManager(int maxPop = 100, int genLength = 1000, int maxCatCount = 3);

    // Initialize with random rodents (optional: clone from best weights)
    void initializePopulation(int count, TerminalMatrix& matrix, const std::vector<double>& bestWeights = {});

    // Initialize cats (optional: clone from best weights)
    void initializeCats(int count, TerminalMatrix& matrix, const std::vector<double>& bestWeights = {});

    // Update all rodents and cats
    void update(TerminalMatrix& matrix);

    // Handle reproduction
    void handleReproduction(TerminalMatrix& matrix);

    // Remove dead rodents
    void removeDeadRodents(TerminalMatrix& matrix);

    // Manage cat population (respawn when they eat or die)
    void manageCats(TerminalMatrix& matrix);

    // Evolution: select best rodents and cats, create new generation
    void evolveGeneration(TerminalMatrix& matrix);

    // Cat evolution
    void evolveCats(TerminalMatrix& matrix);

    // Getters
    int getPopulationSize() const { return population.size(); }
    int getAliveCount() const;
    int getGeneration() const { return generation; }
    double getAverageFitness() const;
    double getBestFitness() const;
    int getCurrentTick() const { return currentTick; }
    int getGenerationLength() const { return generationLength; }
    Rodent* getBestRodent() const;
    Cat* getBestCat() const;
    int getCatCount() const { return cats.size(); }

    // Get population stats
    struct Stats {
        int alive;
        int dead;
        int totalDeaths;
        double avgEnergy;
        double avgFitness;
        double bestFitness;
        int generation;
        int tick;
        int catCount;
        double bestCatFitness;
        int totalRodentsEaten;
    };
    Stats getStats() const;
};

#endif
