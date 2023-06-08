#pragma once

#ifndef Window_C
#define Window_C

#include <SFML/Graphics.hpp>
#include <iostream>

class Window {
    private:
        sf::RenderWindow window;
        std::string title;

        int width, height;

    public:
        Window(int, int, std::string);

        sf::RenderWindow& getWindow();

        void render();
        void update();
};

#endif