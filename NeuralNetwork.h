#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cmath>
#include <random>

class NeuralNetwork {
private:
    std::vector<int> layer_sizes;  // [input_size, hidden1, hidden2, ..., output_size]
    std::vector<std::vector<std::vector<double>>> weights;  // weights[layer][neuron][input]
    std::vector<std::vector<double>> biases;  // biases[layer][neuron]

    // Activation function (sigmoid)
    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // ReLU activation
    double relu(double x) const {
        return x > 0 ? x : 0;
    }

    // Tanh activation
    double tanh_activation(double x) const {
        return std::tanh(x);
    }

public:
    NeuralNetwork(const std::vector<int>& layers);

    // Initialize with random weights
    void randomize(double min = -1.0, double max = 1.0);

    // Set weights from a flat vector (for evolution)
    void setWeights(const std::vector<double>& flat_weights);

    // Get all weights as a flat vector (for evolution)
    std::vector<double> getWeights() const;

    // Get total number of weights
    int getWeightCount() const;

    // Forward pass: input -> output
    std::vector<double> forward(const std::vector<double>& input);

    // Mutate weights for evolution
    void mutate(double mutation_rate, double mutation_amount);

    // Save/load weights to/from file
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);
};

#endif
