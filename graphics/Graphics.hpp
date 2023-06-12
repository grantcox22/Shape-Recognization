#pragma once

#ifndef Window_C
#define Window_C

#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>

#define IMG_SIZE 400

class Window {
    private:
        sf::RenderWindow window;
        std::string title;
        DrawSurface draw_surface;

        int width, height;

    public:
        Window(int, int, std::string);

        sf::RenderWindow& getWindow();

        void render();
        void update();
};

std::vector<double> img_data(std::string src);

class DrawSurface {
    private:
        sf::RectangleShape rect;
        sf::Image pixels;
        std::vector<double> pixel_values;

    public:
        DrawSurface(sf::Vector2f, sf::Vector2f);
        void update();
        void render(sf::RenderTarget&);

};

#endif