#include <iostream>
#include <locale.h>
#include <ncurses.h>
#include "common/include/Menu.h"

// Forward declarations for game entry points
int runEvolutionGame();
int runCityScapeGame();
int runSnakeGame();

int main() {
    // Set locale for UTF-8 support BEFORE ncurses
    setlocale(LC_ALL, "");

    std::cout << "Starting Cursed World...\n";

    // Initialize ncurses
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    clear();

    bool running = true;
    while (running) {
        // Create and show main menu with world icon
        Menu mainMenu("CURSED WORLD - Main Menu", "🌍");
        mainMenu.addOption("Evolution Simulation (Mice vs Cats)", "🧬");
        mainMenu.addOption("City Scape (Urban Navigation)", "🏙️");
        mainMenu.addOption("Snake Game (Classic Gameplay)", "🐍");
        mainMenu.addOption("Exit", "🚪");

        int selection = mainMenu.show();

        // Clear screen after menu selection
        clear();
        refresh();

        // Handle menu selection
        switch (selection) {
            case 0:
                // Launch evolution game
                // Game will handle its own ncurses state
                runEvolutionGame();
                // After game returns, we're back to menu loop
                break;

            case 1:
                // Launch city scape game
                runCityScapeGame();
                break;

            case 2:
                // Launch snake game
                runSnakeGame();
                break;

            case 3:
            case -1:  // ESC
                // Exit
                running = false;
                break;

            default:
                break;
        }
    }

    // Clean up ncurses
    endwin();
    std::cout << "Goodbye!\n";

    return 0;
}

// These functions will be linked from the game object files
// They should NOT call initscr() or endwin() - that's handled by main
extern "C" {
    // Evolution game entry point (defined in games/evolution/src/main.cpp)
    int runEvolutionSimulation();

    // City scape game entry point (defined in games/city_scape/src/CityScapeSimulation.cpp)
    int runCityScapeSimulationDirect();
}

int runEvolutionGame() {
    // Exit ncurses, run the game, then restart ncurses
    endwin();

    // Run the evolution game executable with CUDA backend
    int result = system("cd games/evolution && ./evolution --backend=cuda");

    // Re-initialize ncurses before returning to menu
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    clear();

    return result;
}

int runCityScapeGame() {
    // Exit ncurses, run the game, then restart ncurses
    endwin();

    // Run the city scape game executable
    int result = system("cd games/city_scape && ./city_scape");

    // Re-initialize ncurses before returning to menu
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    clear();

    return result;
}

int runSnakeGame() {
    // Exit ncurses, run the game, then restart ncurses
    endwin();

    // Run the snake game executable
    int result = system("cd games/snake && ./snake");

    // Re-initialize ncurses before returning to menu
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    clear();

    return result;
}
