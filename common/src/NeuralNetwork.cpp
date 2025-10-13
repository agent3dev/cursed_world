#include "../include/NeuralNetwork.h"
#include "../../common/include/Benchmark.h"
#include <fstream>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers) : layer_sizes(layers) {
    // Initialize weights and biases for each layer
    for (size_t i = 0; i < layers.size() - 1; i++) {
        int current_size = layers[i];
        int next_size = layers[i + 1];

        // Weights matrix: [next_layer_neurons][current_layer_neurons]
        std::vector<std::vector<double>> layer_weights(next_size, std::vector<double>(current_size, 0.0));
        weights.push_back(layer_weights);

        // Biases: [next_layer_neurons]
        std::vector<double> layer_biases(next_size, 0.0);
        biases.push_back(layer_biases);
    }

    // Initialize recurrent weights for hidden layer (first hidden layer only)
    if (layers.size() >= 2) {
        int hidden_size = layers[1];  // First hidden layer
        recurrent_weights.resize(hidden_size, std::vector<double>(hidden_size, 0.0));
        hidden_state.resize(hidden_size, 0.0);  // Initialize to zeros
    }
}

void NeuralNetwork::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    for (auto& layer : weights) {
        for (auto& neuron : layer) {
            for (auto& weight : neuron) {
                weight = dis(gen);
            }
        }
    }

    for (auto& layer : biases) {
        for (auto& bias : layer) {
            bias = dis(gen);
        }
    }

    // Randomize recurrent weights
    for (auto& neuron : recurrent_weights) {
        for (auto& weight : neuron) {
            weight = dis(gen);
        }
    }
}

void NeuralNetwork::setWeights(const std::vector<double>& flat_weights) {
    size_t index = 0;

    // Set weights
    for (size_t l = 0; l < weights.size(); l++) {
        for (size_t n = 0; n < weights[l].size(); n++) {
            for (size_t w = 0; w < weights[l][n].size(); w++) {
                if (index < flat_weights.size()) {
                    weights[l][n][w] = flat_weights[index++];
                }
            }
        }
    }

    // Set biases
    for (size_t l = 0; l < biases.size(); l++) {
        for (size_t b = 0; b < biases[l].size(); b++) {
            if (index < flat_weights.size()) {
                biases[l][b] = flat_weights[index++];
            }
        }
    }

    // Set recurrent weights
    for (size_t n = 0; n < recurrent_weights.size(); n++) {
        for (size_t w = 0; w < recurrent_weights[n].size(); w++) {
            if (index < flat_weights.size()) {
                recurrent_weights[n][w] = flat_weights[index++];
            }
        }
    }
}

std::vector<double> NeuralNetwork::getWeights() const {
    std::vector<double> flat_weights;

    // Get all weights
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            for (const auto& weight : neuron) {
                flat_weights.push_back(weight);
            }
        }
    }

    // Get all biases
    for (const auto& layer : biases) {
        for (const auto& bias : layer) {
            flat_weights.push_back(bias);
        }
    }

    // Get recurrent weights
    for (const auto& neuron : recurrent_weights) {
        for (const auto& weight : neuron) {
            flat_weights.push_back(weight);
        }
    }

    return flat_weights;
}

int NeuralNetwork::getWeightCount() const {
    int count = 0;

    // Count weights
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            count += neuron.size();
        }
    }

    // Count biases
    for (const auto& layer : biases) {
        count += layer.size();
    }

    // Count recurrent weights
    for (const auto& neuron : recurrent_weights) {
        count += neuron.size();
    }

    return count;
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    BENCHMARK_SCOPE("NeuralNetwork::forward");

    std::vector<double> current = input;

    // Propagate through each layer
    for (size_t l = 0; l < weights.size(); l++) {
        std::vector<double> next(weights[l].size(), 0.0);

        for (size_t n = 0; n < weights[l].size(); n++) {
            double sum = biases[l][n];

            // Add input from previous layer
            for (size_t i = 0; i < current.size(); i++) {
                sum += current[i] * weights[l][n][i];
            }

            // Add recurrent connections (only for first hidden layer)
            if (l == 0 && !hidden_state.empty()) {
                for (size_t i = 0; i < hidden_state.size(); i++) {
                    sum += hidden_state[i] * recurrent_weights[n][i];
                }
            }

            // Use tanh for all layers
            next[n] = tanh_activation(sum);
        }

        // Store hidden state after first layer (memory)
        if (l == 0 && !recurrent_weights.empty()) {
            hidden_state = next;
        }

        current = next;
    }

    return current;
}

void NeuralNetwork::mutate(double mutation_rate, double mutation_amount) {
    BENCHMARK_SCOPE("NeuralNetwork::mutate");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::normal_distribution<> mutation(0.0, mutation_amount);

    // Mutate weights
    for (auto& layer : weights) {
        for (auto& neuron : layer) {
            for (auto& weight : neuron) {
                if (prob(gen) < mutation_rate) {
                    weight += mutation(gen);
                }
            }
        }
    }

    // Mutate biases
    for (auto& layer : biases) {
        for (auto& bias : layer) {
            if (prob(gen) < mutation_rate) {
                bias += mutation(gen);
            }
        }
    }

    // Mutate recurrent weights
    for (auto& neuron : recurrent_weights) {
        for (auto& weight : neuron) {
            if (prob(gen) < mutation_rate) {
                weight += mutation(gen);
            }
        }
    }
}

void NeuralNetwork::resetHiddenState() {
    // Reset hidden state to zeros
    for (auto& h : hidden_state) {
        h = 0.0;
    }
}

bool NeuralNetwork::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // Save layer sizes first
    file << layer_sizes.size();
    for (int size : layer_sizes) {
        file << " " << size;
    }
    file << "\n";

    // Save all weights
    std::vector<double> flat = getWeights();
    file << flat.size() << "\n";
    for (double w : flat) {
        file << w << " ";
    }
    file << "\n";

    file.close();
    return true;
}

bool NeuralNetwork::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // Read layer sizes
    size_t num_layers;
    file >> num_layers;

    // Sanity check
    if (num_layers == 0 || num_layers > 100 || file.fail()) {
        file.close();
        return false;
    }

    std::vector<int> loaded_layers(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        file >> loaded_layers[i];
        if (file.fail() || loaded_layers[i] <= 0 || loaded_layers[i] > 10000) {
            file.close();
            return false;
        }
    }

    // Verify layer sizes match
    if (loaded_layers != layer_sizes) {
        file.close();
        return false;
    }

    // Read weights
    size_t num_weights;
    file >> num_weights;

    // Sanity check on weight count
    if (num_weights == 0 || num_weights > 1000000 || file.fail()) {
        file.close();
        return false;
    }

    std::vector<double> flat(num_weights);
    for (size_t i = 0; i < num_weights; i++) {
        file >> flat[i];
        if (file.fail()) {
            file.close();
            return false;
        }
    }

    file.close();

    // Set the weights
    setWeights(flat);
    return true;
}
