#include "TerminalMatrix.h"
#include <cstdio>

TerminalMatrix::TerminalMatrix(int w, int h, int dashHeight)
    : width(w), height(h), dashboardHeight(dashHeight), dashboardText("") {
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
    // Draw top border below dashboard and bottom border
    for (int x = 0; x < width; x++) {
        setChar(x, 0, borderChar);  // First line after dashboard
        setChar(x, getHeight() - 1, borderChar);  // Last line
    }
}

void TerminalMatrix::render() const {
    // Render dashboard at the top
    move(0, 0);
    clrtoeol();
    mvaddstr(0, 0, dashboardText.c_str());

    // Render matrix
    for (int y = 0; y < height; y++) {
        int col = 0;  // Track actual terminal column
        for (int x = 0; x < width; x++) {
            const Tile& tile = matrix[y][x];
            const std::string& ch = tile.getChar();

            if (tile.getColorPair() > 0) {
                attron(COLOR_PAIR(tile.getColorPair()));
                mvaddstr(y, col, ch.c_str());
                attroff(COLOR_PAIR(tile.getColorPair()));
            } else {
                mvaddstr(y, col, ch.c_str());
            }

            // Wide characters (emojis) take 2 columns
            if (ch.length() > 1) {
                col += 2;
            } else {
                col += 1;
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
