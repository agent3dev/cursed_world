#include "../include/EvolutionSimulation.h"
#include "../include/PopulationManager.h"
#include "../include/Ghost.h"
#include "../include/Rodent.h"
#include "../include/Cat.h"
#include "../include/NeuralNetwork.h"
#include "../include/config.h"
#include "../../../common/include/TerminalMatrix.h"
#include "../../../common/include/TerrainTypes.h"
#include "../../../common/include/Tile.h"
#include "../../../common/include/Benchmark.h"
#include <ncurses.h>
#include <random>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

EvolutionSimulation::EvolutionSimulation(int argc, char* argv[])
    : Simulation("Cursed World Evolution", 100),  // 100ms frame delay
      wallAnimationInterval(2) {

    // Parse command-line arguments for compute configuration
    computeConfig = std::make_unique<ComputeConfig>();

    if (argc > 0 && argv != nullptr) {
        if (!computeConfig->parseCommandLine(argc, argv)) {
            std::cerr << "[EvolutionSimulation] Warning: Failed to parse command-line arguments\n";
        }
    }

    // Try to load configuration file if it exists
    std::string defaultConfig = ComputeConfig::getDefaultConfigPath();
    if (computeConfig->loadConfig(defaultConfig)) {
        std::cout << "[EvolutionSimulation] Loaded configuration from " << defaultConfig << "\n";
    }
}

void EvolutionSimulation::setComputeConfig(std::unique_ptr<ComputeConfig> config) {
    if (config) {
        computeConfig = std::move(config);
        std::cout << "[EvolutionSimulation] Custom compute configuration set\n";
    }
}

EvolutionSimulation::~EvolutionSimulation() {
    // Destructor defined here where dependent types are complete
}

void EvolutionSimulation::loadBrains() {
    // Try to load best mouse brain from previous run
    std::cout << "Checking for saved mouse brain...\n";
    try {
        NeuralNetwork tempBrain({9, 16, 9});  // Same architecture as rodents
        if (tempBrain.loadFromFile("best_brain.dat")) {
            bestMouseWeights = tempBrain.getWeights();
            std::cout << "[✓] Loaded best mouse brain from previous run - " << bestMouseWeights.size() << " weights\n";
        } else {
            std::cout << "[!] No compatible mouse brain file found - starting fresh\n";
        }
    } catch (const std::exception& e) {
        std::cout << "[!] Failed to load best mouse brain: " << e.what() << " - starting fresh\n";
        bestMouseWeights.clear();
    } catch (...) {
        std::cout << "[!] Failed to load best mouse brain - starting fresh\n";
        bestMouseWeights.clear();
    }

    // Try to load best cat brain from previous run
    std::cout << "Checking for saved cat brain...\n";
    try {
        NeuralNetwork tempCatBrain({10, 16, 9});  // Cat architecture: 10 inputs, 16 hidden, 9 outputs
        if (tempCatBrain.loadFromFile("best_cat_brain.dat")) {
            bestCatWeights = tempCatBrain.getWeights();
            std::cout << "[✓] Loaded best cat brain from previous run - " << bestCatWeights.size() << " weights\n";
        } else {
            std::cout << "[!] No compatible cat brain file found - starting fresh\n";
        }
    } catch (const std::exception& e) {
        std::cout << "[!] Failed to load best cat brain: " << e.what() << " - starting fresh\n";
        bestCatWeights.clear();
    } catch (...) {
        std::cout << "[!] Failed to load best cat brain - starting fresh\n";
        bestCatWeights.clear();
    }
}

void EvolutionSimulation::saveBrains() {
    if (!popManager) return;

    // Save best rodent's brain
    Rodent* bestRodent = popManager->getBestRodent();
    if (bestRodent) {
        if (bestRodent->getBrain().saveToFile("best_brain.dat")) {
            std::cout << "\n[✓] Saved best mouse brain to best_brain.dat\n";
        }
    }

    // Save best cat's brain
    Cat* bestCat = popManager->getBestCat();
    if (bestCat) {
        if (bestCat->getBrain().saveToFile("best_cat_brain.dat")) {
            std::cout << "[✓] Saved best cat brain to best_cat_brain.dat\n";
        }
    }
}

void EvolutionSimulation::initialize() {
    // Load brains before ncurses
    loadBrains();

    // Call base class initialization (sets up ncurses, matrix, calls initializeTerrain)
    Simulation::initialize();

    // Initialize population manager with compute configuration
    popManager = std::make_unique<PopulationManager>(50, 2000, 3, computeConfig.get());  // Max 50 rodents, 2000 ticks per generation, 3 cats

    // Use the brain weights loaded earlier
    popManager->initializePopulation(30, *matrix, bestMouseWeights);  // Start with 30 rodents
    popManager->initializeCats(3, *matrix, bestCatWeights);  // Start with 3 cats

    // Create player-controlled ghost in the center
    int ghostX = matrix->getWidth() / 2;
    int ghostY = matrix->getHeight() / 2;
    playerGhost = std::make_unique<Ghost>(ghostX, ghostY, "👻");
    Tile* ghostTile = matrix->getTile(ghostX, ghostY);
    if (ghostTile) {
        ghostTile->setActuator(playerGhost.get());
    }

    // Initialize wall animation timer
    lastWallToggle = std::chrono::high_resolution_clock::now();
}

void EvolutionSimulation::initializeTerrain() {
    if (!matrix) return;

    // Load config from YAML
    TerrainConfig::Ratios config = TerrainConfig::loadConfig();

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> terrain_dist(0, config.total - 1);
    std::uniform_int_distribution<> char_dist(0, 10);

    // Calculate thresholds from ratios
    const int empty_threshold = config.empty;
    const int seedlings_threshold = empty_threshold + config.seedlings;
    const int dead_trees_threshold = seedlings_threshold + config.dead_trees;

    // Populate grid with terrain types
    for (int y = 1; y < matrix->getHeight() - 1; y++) {
        for (int x = 0; x < matrix->getWidth(); x++) {
            int rand = terrain_dist(gen);
            TerrainType terrain;

            if (rand < empty_threshold) {
                terrain = TerrainType::EMPTY;
            } else if (rand < seedlings_threshold) {
                terrain = TerrainType::SEEDLINGS;
            } else if (rand < dead_trees_threshold) {
                terrain = TerrainType::DEAD_TREES;
            } else {
                terrain = TerrainType::ROCKS;
            }

            // Get character options for this terrain type
            const auto& chars = TERRAIN_CHARS.at(terrain);
            int char_index = char_dist(gen) % chars.size();

            Tile* tile = matrix->getTile(x, y);
            if (tile) {
                tile->setTerrainType(terrain);
                tile->setChar(chars[char_index]);

                // Mark seedlings as edible (food for rodents)
                if (terrain == TerrainType::SEEDLINGS) {
                    tile->setEdible(true);
                }

                // Rocks and dead trees are obstacles (not walkable)
                if (terrain == TerrainType::ROCKS || terrain == TerrainType::DEAD_TREES) {
                    tile->setWalkable(false);
                }
            }
        }
    }

    // Draw border
    matrix->margin("⬛");
}

void EvolutionSimulation::updateEntities() {
    if (!matrix || !popManager) return;

    // Check if it's time to toggle wall animation
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastWallToggle);
    if (elapsed.count() >= wallAnimationInterval) {
        matrix->toggleWallAnimation();
        lastWallToggle = now;
    }

    // Update population
    popManager->update(*matrix);

    // Update seed growth timers
    matrix->updateGrowth();
}

void EvolutionSimulation::renderStats() {
    if (!matrix || !popManager) return;

    auto stats = popManager->getStats();

    // Only update strings every 10 ticks to reduce allocations
    static int lastUpdateTick = -1;
    static int lastUpdateGen = -1;

    if (stats.tick != lastUpdateTick || stats.generation != lastUpdateGen) {
        std::stringstream ss;
        ss << "Gen: " << stats.generation
           << " | Tick: " << stats.tick << "/" << popManager->getGenerationLength()
           << " | Mice: " << stats.alive << "/" << (stats.alive + stats.dead)
           << " | Cats: " << popManager->getCatCount()
           << " | Avg Energy: " << std::fixed << std::setprecision(1) << stats.avgEnergy
           << " | Best Fit: " << std::fixed << std::setprecision(1) << stats.bestFitness;
        matrix->setDashboard(ss.str());

        // Simplified window title - just name and generation
        std::stringstream titleStream;
        titleStream << "Cursed World Evolution | Gen: " << stats.generation;
        matrix->setWindowTitle(titleStream.str());

        lastUpdateTick = stats.tick;
        lastUpdateGen = stats.generation;
    }
}

void EvolutionSimulation::handleGameInput(int ch) {
    if (!playerGhost || !matrix) return;

    // Handle arrow keys for ghost movement
    switch (ch) {
        case KEY_UP:
            playerGhost->move(0, -1, *matrix);
            playerGhost->killNearbyMice(*matrix);
            break;
        case KEY_DOWN:
            playerGhost->move(0, 1, *matrix);
            playerGhost->killNearbyMice(*matrix);
            break;
        case KEY_LEFT:
            playerGhost->move(-1, 0, *matrix);
            playerGhost->killNearbyMice(*matrix);
            break;
        case KEY_RIGHT:
            playerGhost->move(1, 0, *matrix);
            playerGhost->killNearbyMice(*matrix);
            break;
    }
}

void EvolutionSimulation::onQuit() {
    // Save brains before exiting
    saveBrains();

    // Print and save benchmark results
    std::cout << "\n";
    g_benchmark_stats.printReport(std::cout);
    g_benchmark_stats.saveToFile("benchmark_results.txt");
    std::cout << "\n[✓] Benchmark results saved to benchmark_results.txt\n";
}
