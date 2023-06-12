#include "./Network.hpp"

using namespace Network;


// Generates a random value between -1 and 1;
double frand() {
    return (2.0*(double)rand() / RAND_MAX) - 1.0;
}

Neuron::Neuron(int inputs, double bias) : bias{bias} {
    this->weights.resize(inputs+1); // Creates Array with size of inputs + 1, alows for bias term
    generate(weights.begin(), weights.end(), frand); // Fills Weights with random values between -1 and 1
}

double Neuron::run(vector<double> x) {
    x.push_back(bias); // Places bias with in the input of the neuron
    double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0); // Calculates the sum of the products of each input and their weight
    return sigmoid(sum); // Interpolates output to between 0 and 1
}

void Neuron::set_weights(vector<double> w_init) {
    this->weights = w_init;
}

double Neuron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x)); // Interpolates X to between 0 and 1
}

MultiLayerNetwork::MultiLayerNetwork(vector<int> layers, double bias, double alpha) : layers{layers}, bias{bias}, alpha{alpha} {
    for (int i = 0; i < layers.size(); i++) {
        // Initializes Arrays
        values.push_back(vector<double>(layers[i], 0.0));
        d.push_back(vector<double>(layers[i], 0.0));
        network.push_back(vector<Neuron>());
        if (i > 0) for (int j = 0; j < layers[i]; j++)
            network[i].push_back(Neuron(layers[i-1], bias)); // Add neuron with the inputs equal to the number of outputs in the last layer
    }

}

void MultiLayerNetwork::set_weights(vector<vector<vector<double>>> w_init) {
    for (int i = 0; i < w_init.size(); i++) {
        for (int j = 0; j < w_init[i].size(); j++) {
            this->network[i+1][j].set_weights(w_init[i][j]);
        }
    }
}

void MultiLayerNetwork::print_weights() {
    cout << endl;
    for (int i = 1; i < this->network.size(); i++) {
        for (int j = 0; j < this->layers[i]; j++) {
            cout << "Layer " << i << " | Neuron " << j << ": ";
            for (int k = 0; k < this->network[i][j].weights.size(); k++)
                printf("%s%2.4f%s", (this->network[i][j].weights[k] < 0) ? "" : " ", this->network[i][j].weights[k], (k + 1 >= this->network[i][j].weights.size()) ? "" : " | ");
            cout << endl;
        }
    }
    cout << endl;
}

vector<double> MultiLayerNetwork::run(vector<double> x) {
    this->values[0] = x;

    for (int i = 1; i < this->network.size(); i++)
        for (int j = 0; j < this->layers[i]; j++)
            this->values[i][j] = this->network[i][j].run(this->values[i-1]); // Runs each neuron with the outputs of the last layer

    return this->values.back(); // return the final results
}


double MultiLayerNetwork::back_propigation(vector<double> x, vector<double> y) {

    // Step 1 Feed a sample to the network

    vector<double> outputs = this->run(x);

    // Step 2 Calclate the Mean Square Error

    vector<double> error; 
    double MSE = 0.0;
    for (int i = 0; i < y.size(); i++) {
        error.push_back(y[i] - outputs[i]);
        MSE += error[i] * error[i];
    }
    MSE /= this->layers.back();

    // Step 3 Calcuate the Output Error Terms

    for (int i = 0; i < outputs.size(); i++) {
        d.back()[i] = outputs[i] * (1 - outputs[i]) * error[i]; // Dervative of the sigmoid function
    }

    //Step 4 Calculate the Error Term of each Unit on each Layer

    for (int i = this->network.size()-2; i > 0; i--) { // Going Backwards from the output layer
        for (int j = 0; j < this->network[i].size(); j++) {
            double fwd_error = 0.0;
            for (int k = 0; k < this->layers[i+1]; k++)
                fwd_error += this->network[i+1][k].weights[j] * d[i+1][k];
            this->d[i][j] = this->values[i][j] * (1 - this->values[i][j]) * fwd_error;
        }

    }

    // Step 5 and 6 Calculate the Deltas and Update the Weights
    
    for (int i = 1; i < network.size(); i++)
        for (int j = 0; j < layers[i]; j++)
            for (int k = 0; k < layers[i-1]+1; k++) {
                double delta;
                // Alpha slows learning so that the algorithm does not over or under shoot the global minima
                if (k == layers[i-1]) // Checks if the current input is a bias input
                    delta = alpha * this->d[i][j] * this->bias;
                else delta = alpha * this->d[i][j] * this->values[i-1][k];
                this->network[i][j].weights[k] += delta; // Adjusts the weight based off the error term of the current neuron
            }
    return MSE;
}

int MultiLayerNetwork::save_network_to_file(std::string src) {
    ofstream file;
    file.open(SAVES + src);
    if (!file.is_open()) {
        cout << "Failed to open file" << endl;
        return -1;
    }
    file << this->layers.size() << "\n";
    for (auto& layer : layers) file << layer << " ";
    file << endl;
    for (int i = 1; i < this->network.size(); i++) {
        for (int j = 0; j < this->layers[i]; j++) {
            for (int k = 0; k < this->network[i][j].weights.size(); k++)
                file << this->network[i][j].weights[k] << " ";
            file << endl;
        }
    }
    return 0;
}

double MultiLayerNetwork::train_from_img_file(std::string src, int iterations, bool print) {

    ifstream file;
    file.open(TRAINING_DATA + src);
    if (!file.is_open()) {
        cout << "Failed to open file" << endl;
        return 0;
    }

    vector<TrainingData> training_data;

    int num_of_data;
    file >> num_of_data;

    for (int i = 0; i < num_of_data; i++) {
        string src;
        file >> src;
        vector<double> in = img_data(TRAINING_DATA + src);
        vector<double> out(3, 0.0);
        int value;
        file >> value;
        out[value-1] = 1.0;
        training_data.push_back({in, out});
        
    }

    double mse;
    for (int i = 0; i < iterations; i++) {
        mse = 0.0;
        for (int j = 0; j < num_of_data; j++) mse += this->back_propigation(training_data[j].in, training_data[j].out);
        if (print && i % 100 == 0) {
            mse /= num_of_data;
            printf("Gen %d MSE: %0.4f\n", i, mse);
        }
    }

    return mse /= num_of_data;

}

MultiLayerNetwork * Network::create_from_file(std::string src) {
    MultiLayerNetwork * network;
    ifstream file;
    file.open(SAVES + src);
    if (!file.is_open()) {
        cout << "Failed to open file" << endl;
        return NULL;
    }
    int num_layers;
    vector<int> layers;
    file >> num_layers;
    for (int i = 0; i < num_layers; i++) {
        int l = 0;
        file >> l;
        layers.push_back(l);
    }
    network = new MultiLayerNetwork(layers);
    vector<vector<vector<double>>> weights;
    weights.resize(num_layers-1);
    for (int i = 1; i < num_layers; i++) {
        for (int j = 0; j < layers[i]; j++) {
            weights[i-1].push_back(vector<double>());
            for (int k = 0; k < layers[i-1]+1; k++) {
                double w;
                file >> w;
                weights[i-1][j].push_back(w);
            }
        }
    }
    file.close();
    network->set_weights(weights);
    return network;
}

