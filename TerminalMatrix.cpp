#include "TerminalMatrix.h"
#include <cstdio>

TerminalMatrix::TerminalMatrix(int w, int h, int dashHeight)
    : width(w), height(h), dashboardHeight(dashHeight), dashboardText(""), type_view(false), wallAnimationState(false) {
    matrix.resize(height, std::vector<Tile>(width, Tile()));
}

void TerminalMatrix::setDashboard(const std::string& text) {
    dashboardText = text;
}

void TerminalMatrix::setWindowTitle(const std::string& title) {
    // OSC 0 ; title BEL
    printf("\033]0;%s\007", title.c_str());
    fflush(stdout);
}

Tile* TerminalMatrix::getTile(int x, int y) {
    // Offset y by dashboard height
    int actualY = y + dashboardHeight;
    if (isValid(x, actualY)) {
        return &matrix[actualY][x];
    }
    return nullptr;
}

const Tile* TerminalMatrix::getTile(int x, int y) const {
    // Offset y by dashboard height
    int actualY = y + dashboardHeight;
    if (isValid(x, actualY)) {
        return &matrix[actualY][x];
    }
    return nullptr;
}

void TerminalMatrix::setChar(int x, int y, const std::string& c) {
    // Offset y by dashboard height
    int actualY = y + dashboardHeight;
    if (isValid(x, actualY)) {
        matrix[actualY][x].setChar(c);
    }
}

std::string TerminalMatrix::getChar(int x, int y) const {
    // Offset y by dashboard height
    int actualY = y + dashboardHeight;
    if (isValid(x, actualY)) {
        return matrix[actualY][x].getChar();
    }
    return " ";
}

void TerminalMatrix::clear(const std::string& fillChar) {
    for (int y = dashboardHeight; y < height; y++) {
        for (int x = 0; x < width; x++) {
            matrix[y][x].setChar(fillChar);
        }
    }
}

void TerminalMatrix::margin(const std::string& borderChar) {
    // Draw only top and bottom borders (no side walls)
    for (int x = 0; x < width; x++) {
        setChar(x, 0, borderChar);  // First line after dashboard
        setChar(x, getHeight() - 1, borderChar);  // Last line

        // Mark as non-walkable obstacles
        Tile* topTile = getTile(x, 0);
        Tile* bottomTile = getTile(x, getHeight() - 1);
        if (topTile) topTile->setWalkable(false);
        if (bottomTile) bottomTile->setWalkable(false);
    }
}

void TerminalMatrix::render() const {
    // Clear the entire screen first
    erase();

    // Render dashboard at the top
    mvaddstr(0, 0, dashboardText.c_str());

    // Render matrix
    for (int y = 0; y < height; y++) {
        int col = 0;  // Track actual terminal column
        for (int x = 0; x < width; x++) {
            const Tile& tile = matrix[y][x];
            std::string ch;

            // Check if this is a border tile (top or bottom row)
            bool isTopBorder = (y == dashboardHeight);
            bool isBottomBorder = (y == height - 1);
            bool isBorder = isTopBorder || isBottomBorder;

            if (isBorder && !tile.isWalkable()) {
                // Animated border: alternating pattern ⬛⬜⬛⬜
                bool useBlack;
                if (wallAnimationState) {
                    useBlack = (x % 2 == 0);  // Even positions: black
                } else {
                    useBlack = (x % 2 == 1);  // Odd positions: black (swapped)
                }
                ch = useBlack ? "⬛" : "⬜";
            } else if (type_view) {
                // Type view: single character mode
                if (tile.hasActuator()) {
                    if (tile.getActuator()->getType() == ActuatorType::CAT) {
                        ch = "C";
                    } else {
                        ch = "R";
                    }
                } else if (tile.isEdible()) {
                    ch = "F";
                } else if (!tile.isWalkable()) {
                    ch = "O";
                } else {
                    ch = ".";
                }
            } else {
                // Emoji view: display actuator on top if present
                if (tile.hasActuator()) {
                    ch = tile.getActuator()->getChar();
                } else {
                    ch = tile.getChar();
                }
            }

            if (tile.getColorPair() > 0) {
                attron(COLOR_PAIR(tile.getColorPair()));
                mvaddstr(y, col, ch.c_str());
                attroff(COLOR_PAIR(tile.getColorPair()));
            } else {
                mvaddstr(y, col, ch.c_str());
            }

            // In type view, always advance by 1; in emoji view, everything takes 2 columns
            if (type_view) {
                col += 1;
            } else {
                col += 2;  // All tiles take 2 columns (emojis or double-space)
            }
        }
    }
    refresh();
}

void TerminalMatrix::resize(int w, int h) {
    width = w;
    height = h;
    matrix.clear();
    matrix.resize(height, std::vector<Tile>(width, Tile()));
}

bool TerminalMatrix::isValid(int x, int y) const {
    return x >= 0 && x < width && y >= 0 && y < height;
}
