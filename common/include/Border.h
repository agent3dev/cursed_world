#ifndef BORDER_H
#define BORDER_H

#include <string>

// Forward declaration
class TerminalMatrix;

// Border style enumeration
enum class BorderStyle {
    SOLID,          // Single character border (⬛ or custom)
    ANIMATED,       // Alternating pattern (⬛⬜⬛⬜)
    DOUBLE_LINE,    // ╔═╗║╚╝
    SINGLE_LINE,    // ┌─┐│└┘
    CUSTOM          // User-defined characters
};

// Border class for drawing walls/borders around the matrix
class Border {
private:
    BorderStyle style;
    const char* primaryChar;      // Main border character (or top-left corner)
    const char* secondaryChar;    // Secondary char for animated borders (or top-right corner)
    const char* topChar;          // Top edge character (for line styles)
    const char* bottomChar;       // Bottom edge character
    const char* leftChar;         // Left edge character
    const char* rightChar;        // Right edge character
    const char* topLeftChar;      // Top-left corner
    const char* topRightChar;     // Top-right corner
    const char* bottomLeftChar;   // Bottom-left corner
    const char* bottomRightChar;  // Bottom-right corner
    bool isAnimatedBorder;        // Whether border is animated
    bool walkable;                // Whether border tiles are walkable

public:
    // Constructors for different styles
    Border(BorderStyle style = BorderStyle::ANIMATED);
    Border(const char* primary, const char* secondary = nullptr);  // Simple custom border

    // Set custom border characters for box-drawing
    void setCustomBox(
        const char* topLeft, const char* topRight,
        const char* bottomLeft, const char* bottomRight,
        const char* horizontal, const char* vertical
    );

    // Getters
    BorderStyle getStyle() const { return style; }
    const char* getPrimaryChar() const { return primaryChar; }
    const char* getSecondaryChar() const { return secondaryChar; }
    bool isAnimated() const { return isAnimatedBorder; }
    bool isWalkable() const { return walkable; }

    // Setters
    void setWalkable(bool w) { walkable = w; }
    void setAnimated(bool a) { isAnimatedBorder = a; }

    // Get the character for a specific border position
    const char* getCharForPosition(int x, int y, int width, int height, bool animState) const;

    // Draw this border on a matrix
    void draw(TerminalMatrix& matrix) const;

    // Predefined border styles
    static Border solidBlack();
    static Border solidWhite();
    static Border animated();
    static Border doubleLine();
    static Border singleLine();
    static Border custom(const char* ch);
    static Border brick();
};

#endif
