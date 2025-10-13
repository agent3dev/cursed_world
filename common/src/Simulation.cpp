#include "../include/Simulation.h"
#include "../include/TerminalMatrix.h"
#include <ncurses.h>
#include <locale.h>
#include <iostream>

Simulation::Simulation(const std::string& simName, int frameDelay)
    : name(simName), isRunning(false), paused(false),
      tickCount(0), frameDelayMs(frameDelay), matrix(nullptr) {
}

Simulation::~Simulation() {
    // Destructor defined here where TerminalMatrix is complete
}

void Simulation::initializeNcurses() {
    // Set locale for UTF-8/emoji support
    setlocale(LC_ALL, "");

    // Initialize ncurses
    initscr();
    cbreak();              // Disable line buffering
    noecho();              // Don't echo input characters
    keypad(stdscr, TRUE);  // Enable keypad for arrow keys
    curs_set(0);           // Hide cursor
    clear();
}

void Simulation::cleanupNcurses() {
    endwin();
    // Clear the terminal screen for seamless menu return
    std::cout << "\033[2J\033[H";  // ANSI escape codes
}

bool Simulation::handleCommonInput(int ch) {
    switch (ch) {
        case 'q':
        case 'Q':
        case 27:  // ESC key
            onQuit();
            return true;  // Signal to quit

        case ' ':  // Spacebar toggles pause
            paused = !paused;
            if (paused) {
                onPause();
            } else {
                onUnpause();
            }
            break;

        case 't':
        case 'T':  // Toggle type view
            if (matrix) {
                bool current = matrix->getTypeView();
                matrix->setTypeView(!current);
                clear();  // Clear screen to remove artifacts
                if (matrix) {
                    matrix->render();
                }
            }
            break;

        default:
            return false;  // Not handled, let game process it
    }

    return false;  // Continue running
}

void Simulation::initialize() {
    // Get screen dimensions
    int max_y, max_x;
    getmaxyx(stdscr, max_y, max_x);

    // Create terminal matrix with dashboard (1 line for stats)
    matrix = std::make_unique<TerminalMatrix>(max_x / 2, max_y, 1);

    // Set window title
    matrix->setWindowTitle(name);

    // Let derived class initialize terrain
    initializeTerrain();

    // Initial stats render
    renderStats();
}

void Simulation::cleanup() {
    matrix.reset();
}

int Simulation::run() {
    // Initialize ncurses
    initializeNcurses();

    // Initialize simulation
    initialize();
    isRunning = true;

    // Render initial state
    if (matrix) {
        matrix->render();
    }

    // Set non-blocking input
    nodelay(stdscr, TRUE);

    // Main game loop
    int ch;
    while (isRunning) {
        ch = getch();

        // Handle common input (quit, pause, type toggle)
        bool shouldQuit = handleCommonInput(ch);
        if (shouldQuit) {
            break;
        }

        // Let derived class handle game-specific input
        if (ch != ERR) {  // ERR means no input available
            handleGameInput(ch);
        }

        // Update simulation if not paused
        if (!paused) {
            updateEntities();
            renderStats();

            if (matrix) {
                matrix->render();
            }

            tickCount++;

            // Frame rate limiting
            napms(frameDelayMs);
        } else {
            // When paused, sleep to reduce CPU usage
            napms(50);
        }
    }

    // Cleanup
    cleanup();
    cleanupNcurses();

    return 0;
}
