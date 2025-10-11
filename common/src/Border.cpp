#include "../include/Border.h"
#include "../include/TerminalMatrix.h"
#include "../include/Tile.h"

Border::Border(BorderStyle s)
    : style(s), primaryChar("⬛"), secondaryChar("⬜"),
      topChar("─"), bottomChar("─"), leftChar("│"), rightChar("│"),
      topLeftChar("┌"), topRightChar("┐"),
      bottomLeftChar("└"), bottomRightChar("┘"),
      isAnimatedBorder(false), walkable(false) {

    // Set defaults based on style
    switch (style) {
        case BorderStyle::SOLID:
            primaryChar = "⬛";
            secondaryChar = "⬛";
            isAnimatedBorder = false;
            break;

        case BorderStyle::ANIMATED:
            primaryChar = "⬛";
            secondaryChar = "⬜";
            isAnimatedBorder = true;
            break;

        case BorderStyle::DOUBLE_LINE:
            topLeftChar = "╔";
            topRightChar = "╗";
            bottomLeftChar = "╚";
            bottomRightChar = "╝";
            topChar = "═";
            bottomChar = "═";
            leftChar = "║";
            rightChar = "║";
            isAnimatedBorder = false;
            break;

        case BorderStyle::SINGLE_LINE:
            topLeftChar = "┌";
            topRightChar = "┐";
            bottomLeftChar = "└";
            bottomRightChar = "┘";
            topChar = "─";
            bottomChar = "─";
            leftChar = "│";
            rightChar = "│";
            isAnimatedBorder = false;
            break;

        case BorderStyle::CUSTOM:
            // Use defaults, will be set via setCustomBox
            isAnimatedBorder = false;
            break;
    }
}

Border::Border(const char* primary, const char* secondary)
    : style(BorderStyle::CUSTOM), primaryChar(primary),
      secondaryChar(secondary ? secondary : primary),
      topChar(primary), bottomChar(primary),
      leftChar(primary), rightChar(primary),
      topLeftChar(primary), topRightChar(primary),
      bottomLeftChar(primary), bottomRightChar(primary),
      isAnimatedBorder(secondary != nullptr), walkable(false) {
}

void Border::setCustomBox(
    const char* topLeft, const char* topRight,
    const char* bottomLeft, const char* bottomRight,
    const char* horizontal, const char* vertical) {

    style = BorderStyle::CUSTOM;
    topLeftChar = topLeft;
    topRightChar = topRight;
    bottomLeftChar = bottomLeft;
    bottomRightChar = bottomRight;
    topChar = horizontal;
    bottomChar = horizontal;
    leftChar = vertical;
    rightChar = vertical;
}

const char* Border::getCharForPosition(int x, int y, int width, int height, bool animState) const {
    bool isTopLeft = (x == 0 && y == 0);
    bool isTopRight = (x == width - 1 && y == 0);
    bool isBottomLeft = (x == 0 && y == height - 1);
    bool isBottomRight = (x == width - 1 && y == height - 1);
    bool isTop = (y == 0);
    bool isBottom = (y == height - 1);
    bool isLeft = (x == 0);
    bool isRight = (x == width - 1);

    // For line-based borders (double/single/custom box)
    if (style == BorderStyle::DOUBLE_LINE || style == BorderStyle::SINGLE_LINE ||
        (style == BorderStyle::CUSTOM && topLeftChar != primaryChar)) {
        if (isTopLeft) return topLeftChar;
        if (isTopRight) return topRightChar;
        if (isBottomLeft) return bottomLeftChar;
        if (isBottomRight) return bottomRightChar;
        if (isTop) return topChar;
        if (isBottom) return bottomChar;
        if (isLeft) return leftChar;
        if (isRight) return rightChar;
    }

    // For solid/animated borders
    if (isAnimatedBorder && (style == BorderStyle::ANIMATED || secondaryChar != primaryChar)) {
        // Animated pattern
        int pos = (isLeft || isRight) ? y : x;
        bool usePrimary;
        if (animState) {
            usePrimary = (pos % 2 == 0);
        } else {
            usePrimary = (pos % 2 == 1);
        }
        return usePrimary ? primaryChar : secondaryChar;
    }

    // Default: solid border
    return primaryChar;
}

void Border::draw(TerminalMatrix& matrix) const {
    int width = matrix.getWidth();
    int height = matrix.getHeight();

    // Draw top and bottom borders
    for (int x = 0; x < width; x++) {
        const char* topCh = getCharForPosition(x, 0, width, height, false);
        const char* bottomCh = getCharForPosition(x, height - 1, width, height, false);

        matrix.setChar(x, 0, topCh);
        matrix.setChar(x, height - 1, bottomCh);

        // Mark as non-walkable if specified
        if (!walkable) {
            Tile* topTile = matrix.getTile(x, 0);
            Tile* bottomTile = matrix.getTile(x, height - 1);
            if (topTile) topTile->setWalkable(false);
            if (bottomTile) bottomTile->setWalkable(false);
        }
    }

    // Draw left and right borders
    for (int y = 0; y < height; y++) {
        const char* leftCh = getCharForPosition(0, y, width, height, false);
        const char* rightCh = getCharForPosition(width - 1, y, width, height, false);

        matrix.setChar(0, y, leftCh);
        matrix.setChar(width - 1, y, rightCh);

        // Mark as non-walkable if specified
        if (!walkable) {
            Tile* leftTile = matrix.getTile(0, y);
            Tile* rightTile = matrix.getTile(width - 1, y);
            if (leftTile) leftTile->setWalkable(false);
            if (rightTile) rightTile->setWalkable(false);
        }
    }
}

// Static factory methods
Border Border::solidBlack() {
    return Border(BorderStyle::SOLID);
}

Border Border::solidWhite() {
    Border b(BorderStyle::SOLID);
    b.primaryChar = "⬜";
    b.secondaryChar = "⬜";
    return b;
}

Border Border::animated() {
    return Border(BorderStyle::ANIMATED);
}

Border Border::doubleLine() {
    return Border(BorderStyle::DOUBLE_LINE);
}

Border Border::singleLine() {
    return Border(BorderStyle::SINGLE_LINE);
}

Border Border::custom(const char* ch) {
    return Border(ch, nullptr);
}

Border Border::brick() {
    return Border("🧱", nullptr);
}
