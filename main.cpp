#include <ncurses.h>
#include <string>
#include <locale.h>
#include <random>
#include <chrono>
#include <map>
#include <sstream>
#include <iomanip>
#include <iostream>
#include "TerminalMatrix.h"
#include "config.h"
#include "Rodent.h"
#include "PopulationManager.h"
#include "Ghost.h"

int main() {
    std::cout << "Starting Cursed World...\n";

    // Try to load best brain from previous run BEFORE ncurses
    std::vector<double> bestWeights;
    std::cout << "Checking for saved brain...\n";
    try {
        NeuralNetwork tempBrain({9, 16, 9});  // Same architecture as rodents
        if (tempBrain.loadFromFile("best_brain.dat")) {
            bestWeights = tempBrain.getWeights();
            std::cout << "[✓] Loaded best brain from previous run - " << bestWeights.size() << " weights\n";
        } else {
            std::cout << "[!] No compatible brain file found - starting fresh\n";
        }
    } catch (const std::exception& e) {
        std::cout << "[!] Failed to load best brain: " << e.what() << " - starting fresh\n";
        bestWeights.clear();
    } catch (...) {
        std::cout << "[!] Failed to load best brain - starting fresh\n";
        bestWeights.clear();
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Setting locale...\n";
    // Set locale for UTF-8 support
    setlocale(LC_ALL, "");

    std::cout << "Initializing ncurses...\n";
    // Initialize ncurses
    initscr();

    // Disable line buffering
    cbreak();

    // Don't echo input characters
    noecho();

    // Enable keypad for arrow keys
    keypad(stdscr, TRUE);

    // Hide cursor
    curs_set(0);

    // Clear screen
    clear();

    // Get screen dimensions
    int max_y, max_x;
    getmaxyx(stdscr, max_y, max_x);

    // Create terminal matrix with dashboard line for stats
    // Divide max_x by 2 since emojis take 2 columns each
    TerminalMatrix matrix(max_x / 2, max_y, 1);

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
    for (int y = 1; y < matrix.getHeight() - 1; y++) {
        for (int x = 0; x < matrix.getWidth(); x++) {
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

            Tile* tile = matrix.getTile(x, y);
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

    // Draw border using margin method
    matrix.margin("⬛");

    // Calculate loading time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Update window title with loading time
    std::string title = "Cursed World Evolution | Load: " + std::to_string(duration.count()) + "ms";
    matrix.setWindowTitle(title);

    // Initialize population manager
    PopulationManager popManager(150, 2000, 5);  // Max 150 rodents, 2000 ticks per generation, 5 cats

    // Use the brain weights loaded earlier
    popManager.initializePopulation(100, matrix, bestWeights);  // Start with 100 rodents
    popManager.initializeCats(5, matrix);  // Start with 5 cats

    // Create player-controlled ghost in the center
    Ghost playerGhost(max_x / 2, max_y / 2, "👻");
    Tile* ghostTile = matrix.getTile(max_x / 2, max_y / 2);
    if (ghostTile) {
        ghostTile->setActuator(&playerGhost);
    }

    // Render the matrix
    matrix.render();

    // Display mode flags
    bool paused = true;  // Start paused

    // Wall animation timer
    auto lastWallToggle = std::chrono::high_resolution_clock::now();
    const int wallAnimationInterval = 2;  // Toggle every 2 seconds

    // Set non-blocking input
    nodelay(stdscr, TRUE);

    // Main loop - wait for 'q' or ESC to exit
    int ch;
    while (true) {
        ch = getch();
        if (ch == 'q' || ch == 27) {  // 27 is ESC key
            break;
        } else if (ch == ' ') {  // Spacebar toggles pause
            paused = !paused;
        } else if (ch == 't' || ch == 'T') {
            // Toggle type display mode
            bool current = matrix.getTypeView();
            matrix.setTypeView(!current);
            clear();  // Clear the entire screen to remove artifacts
            matrix.render();
        } else if (ch == KEY_UP) {
            playerGhost.move(0, -1, matrix);
            playerGhost.killNearbyMice(matrix);
            matrix.render();
        } else if (ch == KEY_DOWN) {
            playerGhost.move(0, 1, matrix);
            playerGhost.killNearbyMice(matrix);
            matrix.render();
        } else if (ch == KEY_LEFT) {
            playerGhost.move(-1, 0, matrix);
            playerGhost.killNearbyMice(matrix);
            matrix.render();
        } else if (ch == KEY_RIGHT) {
            playerGhost.move(1, 0, matrix);
            playerGhost.killNearbyMice(matrix);
            matrix.render();
        }

        // Check if it's time to toggle wall animation
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastWallToggle);
        if (elapsed.count() >= wallAnimationInterval) {
            matrix.toggleWallAnimation();
            lastWallToggle = now;
        }

        // Update simulation if not paused
        if (!paused) {
            // Update population
            popManager.update(matrix);

            // Update dashboard with stats
            auto stats = popManager.getStats();
            std::stringstream ss;
            ss << "Gen: " << stats.generation
               << " | Tick: " << stats.tick << "/" << popManager.getGenerationLength()
               << " | Mice: " << stats.alive << "/" << (stats.alive + stats.dead)
               << " | Cats: " << popManager.getCatCount()
               << " | Avg Energy: " << std::fixed << std::setprecision(1) << stats.avgEnergy
               << " | Best Fit: " << std::fixed << std::setprecision(1) << stats.bestFitness;
            matrix.setDashboard(ss.str());

            // Update window title with animal counts and tick
            std::stringstream titleStream;
            titleStream << "Cursed World Evolution | Gen: " << stats.generation
                       << " | Tick: " << stats.tick
                       << " | Mice: " << stats.alive
                       << " | Cats: " << popManager.getCatCount();
            matrix.setWindowTitle(titleStream.str());

            matrix.render();

            // Delay for visible updates (slower movement)
            napms(100);
        } else {
            // When paused, sleep a bit to reduce CPU usage
            napms(50);
        }
    }

    // Get final stats before cleanup
    auto finalStats = popManager.getStats();

    // Save best rodent's brain
    Rodent* bestRodent = popManager.getBestRodent();
    if (bestRodent) {
        if (bestRodent->getBrain().saveToFile("best_brain.dat")) {
            std::cout << "\n[Saved best brain to best_brain.dat]\n";
        }
    }

    // Clean up and exit
    endwin();

    // Print summary to standard output
    std::cout << "\n=== CURSED WORLD EVOLUTION SUMMARY ===\n";
    std::cout << "Final Generation: " << finalStats.generation << "\n";
    std::cout << "Final Tick: " << finalStats.tick << "\n";
    std::cout << "Population:\n";
    std::cout << "  - Alive: " << finalStats.alive << "\n";
    std::cout << "  - Total Deaths: " << finalStats.totalDeaths << "\n";
    std::cout << "  - Current Total: " << (finalStats.alive + finalStats.dead) << "\n";
    std::cout << "Performance:\n";
    std::cout << "  - Average Energy: " << std::fixed << std::setprecision(2) << finalStats.avgEnergy << "\n";
    std::cout << "  - Average Fitness: " << std::fixed << std::setprecision(2) << finalStats.avgFitness << "\n";
    std::cout << "  - Best Fitness: " << std::fixed << std::setprecision(2) << finalStats.bestFitness << "\n";
    std::cout << "======================================\n\n";

    return 0;
}
