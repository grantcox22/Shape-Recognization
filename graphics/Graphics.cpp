#include "Graphics.hpp"

Window::Window(int width, int height, std::string title) : width {width}, height {height}, title {title}, draw_surface({50, 50}, {width - 100, height - 100}) {
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
    this->draw_surface.render(this->getWindow());
    this->window.display();
}

std::vector<double> img_data(std::string src) {
    std::vector<double> data;

    sf::Image image;
    image.loadFromFile(src);

    int width = image.getSize().x;
    int height = image.getSize().y;

    data.resize(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            sf::Color c = image.getPixel(x, y);
            double value = ((c.r + c.g + c.b) / 3) / 255;
            data[x + y * width] = value;
        }
    }

    return data;
    
}

DrawSurface::DrawSurface(sf::Vector2f init_pos, sf::Vector2f size) {
    this->rect.setPosition(init_pos);
    this->rect.setSize(size);
    this->rect.setOutlineColor(sf::Color::White);
    this->rect.setOutlineThickness(2.f);

    this->pixels.create(size.x, size.y, sf::Color::Black);
    this->pixel_values.resize(size.x*size.y);
    for (int i = 0; i < this->pixel_values.size(); i++) this->pixel_values[i] = 0.0;
}

void DrawSurface::render(sf::RenderTarget &target) {
    target.draw(this->rect);
}
