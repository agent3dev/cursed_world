#include "../include/TerminalMatrix.h"
#include "../include/Border.h"
#include <cstdio>

TerminalMatrix::TerminalMatrix(int w, int h, int dashHeight)
    : width(w), height(h), dashboardHeight(dashHeight), dashboardText(""),
      type_view(false), wallAnimationState(false) {
    matrix.resize(height, std::vector<Tile>(width, Tile()));
    // Pre-allocate dashboard string capacity to avoid repeated allocations
    dashboardText.reserve(200);
}

void TerminalMatrix::setDashboard(const std::string& text) {
    // Use clear + assign to reuse existing capacity
    dashboardText.clear();
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

void TerminalMatrix::setBorder(Border* b) {
    borderStyle.reset(b);
    if (borderStyle) {
        borderStyle->draw(*this);
    }
}

void TerminalMatrix::margin(const std::string& borderChar) {
    // Legacy function - creates a simple solid border
    Border* b = new Border(borderChar.c_str(), nullptr);
    setBorder(b);
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

            // Check if this is a border tile (top, bottom, left, or right)
            bool isTopBorder = (y == dashboardHeight);
            bool isBottomBorder = (y == height - 1);
            bool isLeftBorder = (x == 0);
            bool isRightBorder = (x == width - 1);
            bool isBorder = isTopBorder || isBottomBorder || isLeftBorder || isRightBorder;

            if (isBorder && !tile.isWalkable() && borderStyle && borderStyle->isAnimated()) {
                // Use border class to get the character for this position
                ch = borderStyle->getCharForPosition(x, y - dashboardHeight, width, getHeight(), wallAnimationState);
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

void TerminalMatrix::updateGrowth() {
    // Update growth timers for all tiles
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            matrix[y][x].tickGrowth();
        }
    }
}
