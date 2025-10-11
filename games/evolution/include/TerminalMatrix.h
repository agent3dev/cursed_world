#ifndef TERMINAL_MATRIX_H
#define TERMINAL_MATRIX_H

#include <vector>
#include <string>
#include <ncurses.h>
#include "Tile.h"

class TerminalMatrix {
private:
    int width;
    int height;
    int dashboardHeight;
    std::vector<std::vector<Tile>> matrix;
    std::string dashboardText;
    bool type_view;  // Toggle between emoji and type display
    bool wallAnimationState;  // Toggle between ⬛ and ⬜ for wall animation

public:
    TerminalMatrix(int w, int h, int dashHeight = 1);

    // Type view mode
    bool getTypeView() const { return type_view; }
    void setTypeView(bool tv) { type_view = tv; }

    // Get dimensions (excluding dashboard)
    int getWidth() const { return width; }
    int getHeight() const { return height - dashboardHeight; }
    int getTotalHeight() const { return height; }
    int getDashboardHeight() const { return dashboardHeight; }

    // Dashboard
    void setDashboard(const std::string& text);
    std::string getDashboard() const { return dashboardText; }

    // Window title
    void setWindowTitle(const std::string& title);

    // Get tile at position (0,0 is below dashboard)
    Tile* getTile(int x, int y);
    const Tile* getTile(int x, int y) const;

    // Set/get character at position (0,0 is below dashboard)
    void setChar(int x, int y, const std::string& c);
    std::string getChar(int x, int y) const;

    // Clear the matrix
    void clear(const std::string& fillChar = " ");

    // Draw a margin/border with specified character
    void margin(const std::string& borderChar = "#");

    // Toggle wall animation state
    void toggleWallAnimation() { wallAnimationState = !wallAnimationState; }

    // Render the matrix to the terminal
    void render() const;

    // Update dimensions (e.g., when terminal is resized)
    void resize(int w, int h);

    // Check if coordinates are valid
    bool isValid(int x, int y) const;

    // Update growth timers for all tiles
    void updateGrowth();
};

#endif
