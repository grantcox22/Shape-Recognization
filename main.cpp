#include <iostream>
#include <SFML/Graphics.hpp>
// #include "./graphics/Window.hpp"
#include "./network/Network.hpp"

int main() {

    // Window w(600, 300, "Image Recognization");
    // while (w.getWindow().isOpen()) {
    //     w.render();
    //     w.update();
    // }

    // And Gate Test
    Network::Neuron n(2);
    n.set_weights({10, 10, -15});
    for (int i = 0; i < 4; i++) {
        double a = i > 1;
        double b = i % 2;
        printf("%0.0f %0.0f | %0.4f\n", a, b, n.run({a, b}));
    }

    return 0;
}