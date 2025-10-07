#include "PopulationManager.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

// Helper to safely create position distributions
static std::uniform_int_distribution<> safeXDist(const TerminalMatrix& matrix) {
    int max_x = std::max(2, matrix.getWidth() - 2);  // Ensure at least (1, 2)
    return std::uniform_int_distribution<>(1, max_x);
}

static std::uniform_int_distribution<> safeYDist(const TerminalMatrix& matrix) {
    int max_y = std::max(2, matrix.getHeight() - 2);  // Ensure at least (1, 2)
    return std::uniform_int_distribution<>(1, max_y);
}

PopulationManager::PopulationManager(int maxPop, int genLength, int maxCatCount)
    : generation(0), maxPopulation(maxPop), generationLength(genLength), currentTick(0), maxCats(maxCatCount), totalDeaths(0) {
    // Initialize debug log
    debugLog.open("death_log.txt", std::ios::app);
    if (debugLog.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        debugLog << "\n=== Simulation started: " << std::ctime(&time);
        debugLog << "Generation,Tick,MouseID,PosX,PosY,Energy,Age,FoodEaten,Cause\n";
    }
}

void PopulationManager::initializePopulation(int count, TerminalMatrix& matrix, const std::vector<double>& bestWeights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    auto x_dist = safeXDist(matrix);
    auto y_dist = safeYDist(matrix);

    for (int i = 0; i < count; i++) {
        int x = x_dist(gen);
        int y = y_dist(gen);

        // Find walkable position
        Tile* tile = matrix.getTile(x, y);
        while (!tile || !tile->isWalkable() || tile->hasActuator()) {
            x = x_dist(gen);
            y = y_dist(gen);
            tile = matrix.getTile(x, y);
        }

        // Create rodent with best weights if available, otherwise random
        auto rodent = std::make_unique<Rodent>(x, y, "🐀", bestWeights);

        // Add some mutation if using best weights (for diversity)
        if (!bestWeights.empty()) {
            rodent->getBrain().mutate(0.2, 0.3);  // 20% mutation rate for initial diversity
        }

        tile->setActuator(rodent.get());
        population.push_back(std::move(rodent));
    }
}

void PopulationManager::initializeCats(int count, TerminalMatrix& matrix, const std::vector<double>& bestWeights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    auto x_dist = safeXDist(matrix);
    auto y_dist = safeYDist(matrix);

    for (int i = 0; i < count; i++) {
        int x = x_dist(gen);
        int y = y_dist(gen);

        // Find walkable position
        Tile* tile = matrix.getTile(x, y);
        int attempts = 0;
        while ((!tile || !tile->isWalkable() || tile->hasActuator()) && attempts < 100) {
            x = x_dist(gen);
            y = y_dist(gen);
            tile = matrix.getTile(x, y);
            attempts++;
        }

        if (tile && tile->isWalkable() && !tile->hasActuator()) {
            // Create cat with best weights if available, otherwise random
            auto cat = std::make_unique<Cat>(x, y, "🐈", bestWeights);

            // Add some mutation if using best weights (for diversity)
            if (!bestWeights.empty()) {
                cat->getBrain().mutate(0.2, 0.3);  // 20% mutation rate for initial diversity
            }

            tile->setActuator(cat.get());
            cats.push_back(std::move(cat));
        }
    }
}

void PopulationManager::manageCats(TerminalMatrix& matrix) {
    // Only maintain cats, don't spawn infinitely
    if (cats.size() >= static_cast<size_t>(maxCats)) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    auto x_dist = safeXDist(matrix);
    auto y_dist = safeYDist(matrix);

    // Try once to spawn a cat
    int x = x_dist(gen);
    int y = y_dist(gen);

    Tile* tile = matrix.getTile(x, y);
    int attempts = 0;
    while ((!tile || !tile->isWalkable() || tile->hasActuator()) && attempts < 50) {
        x = x_dist(gen);
        y = y_dist(gen);
        tile = matrix.getTile(x, y);
        attempts++;
    }

    if (tile && tile->isWalkable() && !tile->hasActuator()) {
        // Spawn new cat with hardcoded AI
        auto cat = std::make_unique<Cat>(x, y, "🐈");
        tile->setActuator(cat.get());
        cats.push_back(std::move(cat));
    }
}

void PopulationManager::update(TerminalMatrix& matrix) {
    currentTick++;

    // Update all cats first (predators move first)
    for (auto& cat : cats) {
        cat->update(matrix);
    }

    // Cats no longer reproduce mid-generation - they evolve between generations
    // This allows for proper fitness evaluation and selection pressure

    // Update all rodents
    for (auto& rodent : population) {
        if (rodent->isAlive()) {
            rodent->update(matrix);
        }
    }

    // Handle reproduction
    handleReproduction(matrix);

    // Remove dead rodents (this also spawns replacements)
    removeDeadRodents(matrix);

    // Manage cat population
    manageCats(matrix);

    // Check if generation is over
    if (currentTick >= generationLength || getAliveCount() == 0) {
        evolveGeneration(matrix);
        currentTick = 0;
    }
}

void PopulationManager::handleReproduction(TerminalMatrix& matrix) {
    if (population.size() >= static_cast<size_t>(maxPopulation)) return;

    std::vector<std::unique_ptr<Rodent>> offspring;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dx_dist(-1, 1);
    std::uniform_int_distribution<> dy_dist(-1, 1);

    int maxOffspringPerTick = 5;  // Limit births per update
    int offspringCount = 0;

    for (auto& rodent : population) {
        // Hard cap to prevent runaway growth
        if (population.size() + offspring.size() >= static_cast<size_t>(maxPopulation)) break;
        if (offspringCount >= maxOffspringPerTick) break;  // Limit per tick

        if (rodent->canReproduce() && population.size() + offspring.size() < static_cast<size_t>(maxPopulation)) {
            // Try to find empty adjacent position (max 3 attempts)
            for (int attempt = 0; attempt < 3; attempt++) {
                int dx = dx_dist(gen);
                int dy = dy_dist(gen);
                int newX = rodent->getX() + dx;
                int newY = rodent->getY() + dy;

                Tile* tile = matrix.getTile(newX, newY);
                if (tile && tile->isWalkable() && !tile->hasActuator()) {
                    Rodent* child = rodent->reproduce(newX, newY);
                    if (child) {
                        tile->setActuator(child);
                        offspring.push_back(std::unique_ptr<Rodent>(child));
                        offspringCount++;
                    }
                    break;  // Success, stop trying
                }
            }
        }
    }

    // Add offspring to population (with size check)
    for (auto& child : offspring) {
        if (population.size() < static_cast<size_t>(maxPopulation)) {
            population.push_back(std::move(child));
        } else {
            break;
        }
    }
}

void PopulationManager::removeDeadRodents(TerminalMatrix& matrix) {
    int deadCount = 0;

    // Count and remove dead rodents
    population.erase(
        std::remove_if(population.begin(), population.end(),
            [&matrix, &deadCount, this](const std::unique_ptr<Rodent>& rodent) {
                if (!rodent->isAlive()) {
                    // Log death details
                    if (debugLog.is_open()) {
                        debugLog << generation << ","
                                 << currentTick << ","
                                 << rodent->getId() << ","
                                 << rodent->getX() << ","
                                 << rodent->getY() << ","
                                 << std::fixed << std::setprecision(2) << rodent->getEnergy() << ","
                                 << rodent->getAge() << ","
                                 << rodent->getFoodEaten() << ","
                                 << (rodent->getEnergy() <= 0.0 ? "Starvation" : "Eaten") << "\n";
                        debugLog.flush();
                    }

                    // Place tombstone and remove from tile
                    Tile* tile = matrix.getTile(rodent->getX(), rodent->getY());
                    if (tile && tile->getActuator() == rodent.get()) {
                        tile->setActuator(nullptr);
                        tile->setChar("🪦");  // Tombstone
                        tile->setWalkable(false);  // Tombstones are obstacles
                    }
                    deadCount++;
                    return true;
                }
                return false;
            }),
        population.end()
    );

    // Track cumulative deaths
    totalDeaths += deadCount;

    // Don't respawn mice - let the generation play out naturally
}

void PopulationManager::evolveCats(TerminalMatrix& matrix) {
    // Sort cats by fitness
    std::sort(cats.begin(), cats.end(),
        [](const std::unique_ptr<Cat>& a, const std::unique_ptr<Cat>& b) {
            return a->getFitness() > b->getFitness();
        });

    // Keep top 50% as parents (more selective than rodents)
    int parentsCount = std::max(2, static_cast<int>(cats.size() * 0.5));
    std::vector<std::unique_ptr<Cat>> parents;

    for (int i = 0; i < parentsCount && i < cats.size(); i++) {
        parents.push_back(std::move(cats[i]));
    }

    // Place skulls for removed cats (bottom 50%)
    for (int i = parentsCount; i < cats.size(); i++) {
        if (cats[i]) {
            Tile* tile = matrix.getTile(cats[i]->getX(), cats[i]->getY());
            if (tile && tile->getActuator() == cats[i].get()) {
                tile->setActuator(nullptr);
                tile->setChar("💀");  // Skull marker for removed cat
                tile->setWalkable(false);  // Skulls are obstacles
            }
        }
    }

    // Clear old cats (including parents, which will be re-added below)
    for (auto& cat : cats) {
        if (cat) {
            Tile* tile = matrix.getTile(cat->getX(), cat->getY());
            if (tile && tile->getActuator() == cat.get()) {
                tile->setActuator(nullptr);
            }
        }
    }
    cats.clear();

    // Create new generation from parents
    std::random_device rd;
    std::mt19937 gen(rd());
    auto x_dist = safeXDist(matrix);
    auto y_dist = safeYDist(matrix);

    // Add parents back with reset positions
    for (auto& parent : parents) {
        int x = x_dist(gen);
        int y = y_dist(gen);
        Tile* tile = matrix.getTile(x, y);

        int attempts = 0;
        while ((!tile || !tile->isWalkable() || tile->hasActuator()) && attempts < 100) {
            x = x_dist(gen);
            y = y_dist(gen);
            tile = matrix.getTile(x, y);
            attempts++;
        }

        if (!tile || !tile->isWalkable() || tile->hasActuator()) {
            continue;
        }

        parent->setPosition(x, y);
        parent->resetRodentsEaten();  // Reset kill counter
        parent->resetMemory();  // Reset recurrent memory
        tile->setActuator(parent.get());
        cats.push_back(std::move(parent));
    }

    // Create offspring from parents to reach initial cat count
    int targetCats = maxCats;
    int offspringCount = targetCats - cats.size();

    if (offspringCount < 0) offspringCount = 0;
    if (cats.empty()) return;  // Can't create offspring without parents

    std::uniform_int_distribution<> parent_dist(0, cats.size() - 1);

    for (int i = 0; i < offspringCount && cats.size() < static_cast<size_t>(maxCats); i++) {
        int parentIdx = parent_dist(gen);
        std::vector<double> parentWeights = cats[parentIdx]->getBrain().getWeights();

        int x = x_dist(gen);
        int y = y_dist(gen);
        Tile* tile = matrix.getTile(x, y);

        int attempts = 0;
        while ((!tile || !tile->isWalkable() || tile->hasActuator()) && attempts < 100) {
            x = x_dist(gen);
            y = y_dist(gen);
            tile = matrix.getTile(x, y);
            attempts++;
        }

        if (!tile || !tile->isWalkable() || tile->hasActuator()) {
            continue;
        }

        auto offspring = std::make_unique<Cat>(x, y, "🐈", parentWeights);
        offspring->getBrain().mutate(0.15, 0.4);  // 15% mutation rate for cats
        tile->setActuator(offspring.get());
        cats.push_back(std::move(offspring));
    }
}

void PopulationManager::evolveGeneration(TerminalMatrix& matrix) {
    generation++;

    // Evolve cats alongside rodents
    evolveCats(matrix);

    // Clear all tombstones and skulls from the map
    for (int y = 0; y < matrix.getHeight(); y++) {
        for (int x = 0; x < matrix.getWidth(); x++) {
            Tile* tile = matrix.getTile(x, y);
            if (tile && (tile->getChar() == "🪦" || tile->getChar() == "💀")) {
                tile->setChar("  ");  // Reset to empty
                tile->setWalkable(true);  // Make walkable again
                tile->setTerrainType(TerrainType::EMPTY);
            }
        }
    }

    // Sort by fitness
    std::sort(population.begin(), population.end(),
        [](const std::unique_ptr<Rodent>& a, const std::unique_ptr<Rodent>& b) {
            return a->getFitness() > b->getFitness();
        });

    // Keep top 20% as parents
    int parentsCount = std::max(2, static_cast<int>(population.size() * 0.2));
    std::vector<std::unique_ptr<Rodent>> parents;

    for (int i = 0; i < parentsCount && i < population.size(); i++) {
        parents.push_back(std::move(population[i]));
    }

    // If everyone died, restart with fresh population
    if (parents.empty()) {
        // Clear old population
        for (auto& rodent : population) {
            if (rodent) {
                Tile* tile = matrix.getTile(rodent->getX(), rodent->getY());
                if (tile && tile->getActuator() == rodent.get()) {
                    tile->setActuator(nullptr);
                }
            }
        }
        population.clear();

        // Initialize fresh random population
        initializePopulation(5, matrix);
        return;
    }

    // Clear old population
    for (auto& rodent : population) {
        if (rodent) {
            Tile* tile = matrix.getTile(rodent->getX(), rodent->getY());
            if (tile && tile->getActuator() == rodent.get()) {
                tile->setActuator(nullptr);
            }
        }
    }
    population.clear();

    // Create new generation from parents
    std::random_device rd;
    std::mt19937 gen(rd());
    auto x_dist = safeXDist(matrix);
    auto y_dist = safeYDist(matrix);

    // Add parents back
    for (auto& parent : parents) {
        // Reset parent state
        int x = x_dist(gen);
        int y = y_dist(gen);
        Tile* tile = matrix.getTile(x, y);

        // Safety: max 100 tries to find empty tile
        int attempts = 0;
        while ((!tile || !tile->isWalkable() || tile->hasActuator()) && attempts < 100) {
            x = x_dist(gen);
            y = y_dist(gen);
            tile = matrix.getTile(x, y);
            attempts++;
        }

        // If we couldn't find a spot, skip this parent
        if (!tile || !tile->isWalkable() || tile->hasActuator()) {
            continue;
        }

        parent->setPosition(x, y);
        tile->setActuator(parent.get());
        population.push_back(std::move(parent));
    }

    // Create offspring from parents
    int targetPopulation = 80;
    int offspringCount = targetPopulation - population.size();  // Target population of 80

    // Safety check
    if (offspringCount < 0) offspringCount = 0;
    if (offspringCount > maxPopulation) offspringCount = maxPopulation - population.size();

    // Create parent distribution only if we have parents
    if (population.empty()) return;  // Can't create offspring without parents
    std::uniform_int_distribution<> parent_dist(0, population.size() - 1);

    for (int i = 0; i < offspringCount && population.size() < static_cast<size_t>(maxPopulation); i++) {
        int parentIdx = parent_dist(gen);
        std::vector<double> parentWeights = population[parentIdx]->getBrain().getWeights();

        int x = x_dist(gen);
        int y = y_dist(gen);
        Tile* tile = matrix.getTile(x, y);

        // Safety: max 100 tries to find empty tile
        int attempts = 0;
        while ((!tile || !tile->isWalkable() || tile->hasActuator()) && attempts < 100) {
            x = x_dist(gen);
            y = y_dist(gen);
            tile = matrix.getTile(x, y);
            attempts++;
        }

        // If we couldn't find a spot, skip this offspring
        if (!tile || !tile->isWalkable() || tile->hasActuator()) {
            continue;
        }

        auto offspring = std::make_unique<Rodent>(x, y, "🐀", parentWeights);
        offspring->getBrain().mutate(0.1, 0.5);  // 10% mutation rate
        tile->setActuator(offspring.get());
        population.push_back(std::move(offspring));
    }
}

int PopulationManager::getAliveCount() const {
    int count = 0;
    for (const auto& rodent : population) {
        if (rodent->isAlive()) count++;
    }
    return count;
}

double PopulationManager::getAverageFitness() const {
    if (population.empty()) return 0.0;

    double total = 0.0;
    for (const auto& rodent : population) {
        total += rodent->getFitness();
    }
    return total / population.size();
}

double PopulationManager::getBestFitness() const {
    if (population.empty()) return 0.0;

    double best = 0.0;
    for (const auto& rodent : population) {
        if (rodent->getFitness() > best) {
            best = rodent->getFitness();
        }
    }
    return best;
}

PopulationManager::Stats PopulationManager::getStats() const {
    Stats stats;
    stats.alive = getAliveCount();
    stats.dead = population.size() - stats.alive;
    stats.totalDeaths = totalDeaths;
    stats.generation = generation;
    stats.tick = currentTick;

    double totalEnergy = 0.0;
    double totalFitness = 0.0;
    double bestFit = 0.0;

    for (const auto& rodent : population) {
        if (rodent->isAlive()) {
            totalEnergy += rodent->getEnergy();
        }
        totalFitness += rodent->getFitness();
        if (rodent->getFitness() > bestFit) {
            bestFit = rodent->getFitness();
        }
    }

    stats.avgEnergy = stats.alive > 0 ? totalEnergy / stats.alive : 0.0;
    stats.avgFitness = population.size() > 0 ? totalFitness / population.size() : 0.0;
    stats.bestFitness = bestFit;

    // Cat statistics
    stats.catCount = cats.size();
    stats.bestCatFitness = 0.0;
    stats.totalRodentsEaten = 0;

    for (const auto& cat : cats) {
        stats.totalRodentsEaten += cat->getRodentsEaten();
        if (cat->getFitness() > stats.bestCatFitness) {
            stats.bestCatFitness = cat->getFitness();
        }
    }

    return stats;
}

Rodent* PopulationManager::getBestRodent() const {
    if (population.empty()) return nullptr;

    Rodent* best = nullptr;
    double bestFitness = 0.0;

    for (const auto& rodent : population) {
        if (rodent->getFitness() > bestFitness) {
            bestFitness = rodent->getFitness();
            best = rodent.get();
        }
    }

    return best;
}

Cat* PopulationManager::getBestCat() const {
    if (cats.empty()) return nullptr;

    Cat* best = nullptr;
    double bestFitness = 0.0;

    for (const auto& cat : cats) {
        if (cat->getFitness() > bestFitness) {
            bestFitness = cat->getFitness();
            best = cat.get();
        }
    }

    return best;
}
