#ifndef MENU_H
#define MENU_H

#include <ncurses.h>
#include <string>
#include <vector>

struct MenuOption {
    std::string text;
    std::string icon;  // Emoji or character icon

    MenuOption(const std::string& t, const std::string& i = "")
        : text(t), icon(i) {}
};

class Menu {
private:
    std::vector<MenuOption> options;
    int currentSelection;
    std::string title;
    std::string titleIcon;  // Icon for the menu title
    int maxY, maxX;

public:
    Menu(const std::string& menuTitle, const std::string& icon = "");
    ~Menu();

    // Add menu option with text only (backward compatible)
    void addOption(const std::string& option);

    // Add menu option with icon
    void addOption(const std::string& option, const std::string& icon);

    // Set title icon
    void setTitleIcon(const std::string& icon) { titleIcon = icon; }

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
