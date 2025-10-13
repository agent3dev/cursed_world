#include "../include/CPUBackend.h"
#include "../include/NeuralNetwork.h"
#include <cmath>
#include <algorithm>
#include <iostream>

CPUBackend::CPUBackend()
    : initialized_(false), total_time_ms_(0.0), operation_count_(0) {
}

CPUBackend::~CPUBackend() {
    cleanup();
}

bool CPUBackend::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "[CPUBackend] Initializing CPU compute backend...\n";
    initialized_ = true;
    resetStats();
    return true;
}

void CPUBackend::cleanup() {
    if (!initialized_) {
        return;
    }

    std::cout << "[CPUBackend] Cleaning up CPU backend...\n";
    initialized_ = false;
}

void CPUBackend::batchedForward(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<NeuralNetwork*>& networks,
    std::vector<std::vector<double>>& outputs
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Resize output vector to match input count
    outputs.resize(inputs.size());

    // Process each neural network independently
    for (size_t i = 0; i < inputs.size(); i++) {
        if (i < networks.size() && networks[i]) {
            outputs[i] = networks[i]->forward(inputs[i]);
        } else {
            // No network available - return zeros
            outputs[i] = std::vector<double>(9, 0.0);  // 9 outputs (8 directions + stay)
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_ms_ += duration.count() / 1000.0;
    operation_count_++;
}

void CPUBackend::findNearestEntities(
    const std::vector<AgentPosition>& agent_positions,
    const std::vector<AgentPosition>& target_positions,
    std::vector<NearestEntityResult>& results,
    int search_radius
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Resize results to match agent count
    results.resize(agent_positions.size());

    // For each agent, find the nearest target
    for (size_t i = 0; i < agent_positions.size(); i++) {
        NearestEntityResult& result = results[i];
        result.found = false;
        result.distance = 999999;

        const AgentPosition& agent = agent_positions[i];

        // Search for nearest target within radius
        for (const AgentPosition& target : target_positions) {
            int dx = target.x - agent.x;
            int dy = target.y - agent.y;

            // Skip if outside search radius (using Manhattan distance)
            if (std::abs(dx) > search_radius || std::abs(dy) > search_radius) {
                continue;
            }

            int distance = std::abs(dx) + std::abs(dy);

            // Skip if not closer than current best
            if (distance >= result.distance) {
                continue;
            }

            // Found a closer target
            result.dx = dx;
            result.dy = dy;
            result.distance = distance;
            result.found = true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_ms_ += duration.count() / 1000.0;
    operation_count_++;
}

void CPUBackend::batchedMutate(
    const std::vector<NeuralNetwork*>& networks,
    double mutation_rate,
    double mutation_amount
) {
    auto start = std::chrono::high_resolution_clock::now();

    // Mutate each network independently
    for (NeuralNetwork* network : networks) {
        if (network) {
            network->mutate(mutation_rate, mutation_amount);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_ms_ += duration.count() / 1000.0;
    operation_count_++;
}
