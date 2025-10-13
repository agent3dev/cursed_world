#ifndef CPU_BACKEND_H
#define CPU_BACKEND_H

#include "ComputeBackend.h"
#include <chrono>

/**
 * CPU-based compute backend
 *
 * This backend is always available and requires no special hardware.
 * It uses standard C++ to perform all computations on the CPU.
 *
 * Performance characteristics:
 * - No GPU transfer overhead
 * - Good for small populations (<100 agents)
 * - Straightforward debugging
 * - Linear scaling with population size
 */
class CPUBackend : public ComputeBackend {
private:
    bool initialized_;

    // Performance tracking
    double total_time_ms_;
    int operation_count_;

public:
    CPUBackend();
    virtual ~CPUBackend();

    // ComputeBackend interface implementation
    BackendType getType() const override { return BackendType::CPU; }
    std::string getName() const override { return "CPU"; }

    bool initialize() override;
    void cleanup() override;
    bool isAvailable() const override { return true; }  // CPU always available

    void batchedForward(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<NeuralNetwork*>& networks,
        std::vector<std::vector<double>>& outputs
    ) override;

    void findNearestEntities(
        const std::vector<AgentPosition>& agent_positions,
        const std::vector<AgentPosition>& target_positions,
        std::vector<NearestEntityResult>& results,
        int search_radius
    ) override;

    void batchedMutate(
        const std::vector<NeuralNetwork*>& networks,
        double mutation_rate,
        double mutation_amount
    ) override;

    void getStats(double& total_time_ms, int& operation_count) const override {
        total_time_ms = total_time_ms_;
        operation_count = operation_count_;
    }

    void resetStats() override {
        total_time_ms_ = 0.0;
        operation_count_ = 0;
    }
};

#endif // CPU_BACKEND_H
