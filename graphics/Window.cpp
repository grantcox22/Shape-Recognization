#include "Window.hpp"

Window::Window(int width, int height, std::string title) : width {width}, height {height}, title {title} {
    this->window.create(sf::VideoMode(width, height), title, sf::Style::Titlebar | sf::Style::Close);
}

sf::RenderWindow & Window::getWindow() {
    return this->window;
}

void Window::update() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        }
    }
}

void Window::render() {
    this->window.clear();

    this->window.display();
}