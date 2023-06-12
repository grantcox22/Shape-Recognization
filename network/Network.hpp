#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

#include "../graphics/Graphics.hpp"

#define SAVES "src/saves/"
#define TRAINING_DATA "src/training data/"

using namespace std;

namespace Network {
    class Neuron {
        public:
            vector<double> weights; // Weights for each input
            double bias; // Bias
        
            Neuron(int inputs, double bias=1.0);
            double run(vector<double> x); // "Fires" Neuron
            void set_weights(vector<double> w_init); // Initalizes the weights
            double sigmoid(double x); // Normalization Function
    };

    class MultiLayerNetwork {
        private:
            vector<int> layers; // Number of Neurons per layer
            double bias; // Bias
            double alpha; // Learning Rate

            vector<vector<Neuron>> network; // Neurons
            vector<vector<double>> values; // Cache
            vector<vector<double>> d; // Error Terms

        public:
            MultiLayerNetwork(vector<int> layers, double bias=1.0, double alpha=0.5);
            void set_weights(vector<vector<vector<double>>> w_init); // Init Weights
            void print_weights(); // Prints Weights for each Neuron in the Network
            vector<double> run(vector<double> x); // Runs the network
            double back_propigation(vector<double> x, vector<double> y); // Trains Network

            int save_network_to_file(std::string src); // Saves Network to File

            double train_from_img_file(std::string src, int iterations, bool print=false);

    };
    
    MultiLayerNetwork * create_from_file(std::string src); // Loads Network from File

    struct TrainingData {
        vector<double> in;
        vector<double> out;
    } typedef TrainingData;

};