#ifndef MENU_H
#define MENU_H

#include <ncurses.h>
#include <string>
#include <vector>

class Menu {
private:
    std::vector<std::string> options;
    int currentSelection;
    std::string title;
    int maxY, maxX;

public:
    Menu(const std::string& menuTitle);
    ~Menu();

    // Add menu option
    void addOption(const std::string& option);

    // Display menu and handle input
    // Returns the index of the selected option (0-based)
    int show();

    // Clear and redraw menu
    void render();

private:
    void drawBorder();
    void drawOptions();
    void drawTitle();
};

#endif
