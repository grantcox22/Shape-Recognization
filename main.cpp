#include <iostream>
// #include <SFML/Graphics.hpp>
// #include "./graphics/Graphics.hpp"
#include "./network/Network.hpp"

int main() {

    srand(time(NULL));
    rand();

    Network::MultiLayerNetwork * image_recognizer = Network::create_from_file("save.txt");
    // double mse = image_recognizer->train_from_img_file("training.txt", 10000, true);
    // printf("Final MSE: %0.10f\n", mse);
    image_recognizer->print_weights();
    image_recognizer->save_network_to_file("save.txt");

    Window w(500, 500, "Image Recognization");
    while (w.getWindow().isOpen()) {
        w.render();
        w.update();
    }

    return 0;
}