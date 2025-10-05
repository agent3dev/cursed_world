#include <ncurses.h>
#include <string>
#include <locale.h>
#include <random>
#include <chrono>
#include <map>
#include "TerminalMatrix.h"
#include "config.h"
#include "Rodent.h"

int main() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set locale for UTF-8 support
    setlocale(LC_ALL, "");

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

    // Create terminal matrix (no dashboard line)
    TerminalMatrix matrix(max_x, max_y, 0);

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
    std::string title = "Cursed World | Load: " + std::to_string(duration.count()) + "ms";
    matrix.setWindowTitle(title);

    // Spawn a rodent in the middle of the map
    int rodent_x = matrix.getWidth() / 2;
    int rodent_y = matrix.getHeight() / 2;
    Rodent rodent(rodent_x, rodent_y, "🐀");

    // Store original characters for toggle
    std::map<std::pair<int, int>, std::string> original_chars;
    for (int y = 1; y < matrix.getHeight() - 1; y++) {
        for (int x = 0; x < matrix.getWidth(); x++) {
            Tile* tile = matrix.getTile(x, y);
            if (tile) {
                original_chars[{x, y}] = tile->getChar();
            }
        }
    }

    // Render the matrix with rodent
    Tile* rodent_tile = matrix.getTile(rodent.getX(), rodent.getY());
    std::string rodent_original = rodent_tile ? rodent_tile->getChar() : " ";
    if (rodent_tile) {
        rodent_tile->setChar(rodent.getChar());
    }
    matrix.render();

    // Display mode flags
    bool show_types = false;
    bool paused = true;  // Start paused

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
            show_types = !show_types;

            // Update display
            for (int y = 1; y < matrix.getHeight() - 1; y++) {
                for (int x = 0; x < matrix.getWidth(); x++) {
                    Tile* tile = matrix.getTile(x, y);
                    if (tile) {
                        if (show_types) {
                            // Show type as letter
                            TerrainType terrain = tile->getTerrainType();
                            if (terrain == TerrainType::EMPTY) {
                                // Keep empty as is
                            } else if (terrain == TerrainType::PLANTS) {
                                tile->setChar("P");
                            } else if (terrain == TerrainType::SEEDLINGS) {
                                tile->setChar(tile->isEdible() ? "F" : "f");  // F for Food
                            } else if (terrain == TerrainType::DEAD_TREES) {
                                tile->setChar("O");  // O for obstacle
                            } else if (terrain == TerrainType::ROCKS) {
                                tile->setChar("O");  // O for obstacle
                            }
                        } else {
                            // Restore original emoji
                            tile->setChar(original_chars[{x, y}]);
                        }
                    }
                }
            }

            // Re-render rodent on top
            rodent_tile = matrix.getTile(rodent.getX(), rodent.getY());
            if (rodent_tile) {
                if (show_types) {
                    rodent_tile->setChar("R");  // R for Rodent in type mode
                } else {
                    rodent_tile->setChar(rodent.getChar());
                }
            }

            matrix.render();
        }

        // Update simulation if not paused
        if (!paused) {
            // Restore terrain at old rodent position
            rodent_tile = matrix.getTile(rodent.getX(), rodent.getY());
            if (rodent_tile) {
                if (show_types) {
                    TerrainType terrain = rodent_tile->getTerrainType();
                    if (terrain == TerrainType::SEEDLINGS && rodent_tile->isEdible()) {
                        rodent_tile->setChar("F");
                    } else if (terrain == TerrainType::DEAD_TREES || terrain == TerrainType::ROCKS) {
                        rodent_tile->setChar("O");
                    } else if (terrain == TerrainType::PLANTS) {
                        rodent_tile->setChar("P");
                    } else {
                        rodent_tile->setChar(original_chars[{rodent.getX(), rodent.getY()}]);
                    }
                } else {
                    rodent_tile->setChar(original_chars[{rodent.getX(), rodent.getY()}]);
                }
            }

            // Update rodent AI
            rodent.update(matrix);

            // Update original_chars if rodent ate something
            rodent_tile = matrix.getTile(rodent.getX(), rodent.getY());
            if (rodent_tile) {
                original_chars[{rodent.getX(), rodent.getY()}] = rodent_tile->getChar();
            }

            // Draw rodent at new position
            if (rodent_tile) {
                if (show_types) {
                    rodent_tile->setChar("R");
                } else {
                    rodent_tile->setChar(rodent.getChar());
                }
            }

            matrix.render();

            // Delay for visible updates (60 FPS)
            napms(16);
        } else {
            // When paused, sleep a bit to reduce CPU usage
            napms(50);
        }
    }

    // Clean up and exit
    endwin();

    return 0;
}
