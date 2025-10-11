#include "../include/Menu.h"
#include <algorithm>

Menu::Menu(const std::string& menuTitle)
    : currentSelection(0), title(menuTitle) {
    // Constructor - ncurses will be initialized by caller
}

Menu::~Menu() {
    // Destructor - ncurses cleanup handled by caller
}

void Menu::addOption(const std::string& option) {
    options.push_back(option);
}

void Menu::drawBorder() {
    int boxWidth = 60;
    int boxHeight = static_cast<int>(options.size()) + 8;
    int startY = (maxY - boxHeight) / 2;
    int startX = (maxX - boxWidth) / 2;

    // Draw border
    attron(A_BOLD);
    for (int i = 0; i < boxWidth; i++) {
        mvprintw(startY, startX + i, "═");
        mvprintw(startY + boxHeight - 1, startX + i, "═");
    }
    for (int i = 1; i < boxHeight - 1; i++) {
        mvprintw(startY + i, startX, "║");
        mvprintw(startY + i, startX + boxWidth - 1, "║");
    }
    mvprintw(startY, startX, "╔");
    mvprintw(startY, startX + boxWidth - 1, "╗");
    mvprintw(startY + boxHeight - 1, startX, "╚");
    mvprintw(startY + boxHeight - 1, startX + boxWidth - 1, "╝");
    attroff(A_BOLD);
}

void Menu::drawTitle() {
    int boxWidth = 60;
    int boxHeight = static_cast<int>(options.size()) + 8;
    int startY = (maxY - boxHeight) / 2;
    int startX = (maxX - boxWidth) / 2;

    // Draw title centered
    int titleX = startX + (boxWidth - static_cast<int>(title.length())) / 2;
    attron(A_BOLD | A_UNDERLINE);
    mvprintw(startY + 2, titleX, "%s", title.c_str());
    attroff(A_BOLD | A_UNDERLINE);
}

void Menu::drawOptions() {
    int boxWidth = 60;
    int boxHeight = static_cast<int>(options.size()) + 8;
    int startY = (maxY - boxHeight) / 2;
    int startX = (maxX - boxWidth) / 2;

    // Draw options
    for (size_t i = 0; i < options.size(); i++) {
        int optionY = startY + 5 + static_cast<int>(i);
        int optionX = startX + 5;

        if (static_cast<int>(i) == currentSelection) {
            attron(A_REVERSE | A_BOLD);
            mvprintw(optionY, optionX, "> %s", options[i].c_str());
            attroff(A_REVERSE | A_BOLD);
        } else {
            mvprintw(optionY, optionX, "  %s", options[i].c_str());
        }
    }

    // Draw instructions
    int instructY = startY + boxHeight - 3;
    mvprintw(instructY, startX + 5, "Use UP/DOWN arrows to navigate, ENTER to select");
}

void Menu::render() {
    clear();
    getmaxyx(stdscr, maxY, maxX);
    drawBorder();
    drawTitle();
    drawOptions();
    refresh();
}

int Menu::show() {
    // Set up ncurses for menu
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    render();

    int ch;
    while (true) {
        ch = getch();

        switch (ch) {
            case KEY_UP:
                currentSelection = (currentSelection - 1 + static_cast<int>(options.size())) % static_cast<int>(options.size());
                render();
                break;

            case KEY_DOWN:
                currentSelection = (currentSelection + 1) % static_cast<int>(options.size());
                render();
                break;

            case 10: // ENTER key
            case KEY_ENTER:
                return currentSelection;

            case 27: // ESC key
            case 'q':
            case 'Q':
                return -1; // Exit signal
        }
    }
}
