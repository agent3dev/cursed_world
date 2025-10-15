#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "ComputeBackend.h"

// Only compile CUDA backend if USE_CUDA is defined
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <chrono>

/**
 * CUDA-based compute backend
 *
 * This backend requires CUDA-capable GPU and CUDA toolkit.
 * It accelerates compute-intensive operations by running them on the GPU.
 *
 * Performance characteristics:
 * - GPU transfer overhead (significant for small batches)
 * - Excellent for large populations (>200 agents)
 * - Parallel processing of all agents simultaneously
 * - O(1) scaling for operations within GPU capacity
 *
 * Build requirements:
 * - NVIDIA GPU with compute capability >= 6.0
 * - CUDA toolkit 11.0 or later
 * - Compile with: make ENABLE_CUDA=1
 */
class CUDABackend : public ComputeBackend {
private:
    bool initialized_;
    int device_id_;
    cudaDeviceProp device_properties_;

    // GPU memory buffers
    double* d_inputs_;
    double* d_outputs_;
    double* d_weights1_;  // Layer 1 weights
    double* d_biases1_;   // Layer 1 biases
    double* d_weights2_;  // Layer 2 weights
    double* d_biases2_;   // Layer 2 biases
    double* d_recurrent_; // Recurrent weights
    double* d_hidden_state_; // Previous hidden state
    double* d_hidden_output_; // New hidden state

    // For distance search
    int* d_agent_x_;
    int* d_agent_y_;
    int* d_target_x_;
    int* d_target_y_;
    int* d_result_dx_;
    int* d_result_dy_;
    int* d_result_distance_;
    int* d_result_found_;

    // For mutation
    double* d_weights_all_;
    double* d_random_prob_;
    double* d_random_amount_;

    size_t allocated_size_;
    size_t max_batch_size_;  // Maximum batch size allocated for

    // Performance tracking
    double total_time_ms_;
    int operation_count_;

    // Helper methods
    bool checkCudaDevice();
    bool allocateGPUMemory(size_t size);
    void freeGPUMemory();

public:
    CUDABackend();
    virtual ~CUDABackend();

    // ComputeBackend interface implementation
    BackendType getType() const override { return BackendType::CUDA; }
    std::string getName() const override;

    bool initialize() override;
    void cleanup() override;
    bool isAvailable() const override;

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

    // CUDA-specific methods
    int getDeviceId() const { return device_id_; }
    const cudaDeviceProp& getDeviceProperties() const { return device_properties_; }
};

#else  // !USE_CUDA

/**
 * Stub CUDA backend for when CUDA is not available
 *
 * This allows the code to compile without CUDA support.
 * All methods will return failure/unavailable status.
 */
class CUDABackend : public ComputeBackend {
public:
    CUDABackend() {}
    virtual ~CUDABackend() {}

    BackendType getType() const override { return BackendType::CUDA; }
    std::string getName() const override { return "CUDA (Not Available)"; }

    bool initialize() override { return false; }
    void cleanup() override {}
    bool isAvailable() const override { return false; }

    void batchedForward(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<NeuralNetwork*>& networks,
        std::vector<std::vector<double>>& outputs
    ) override {
        // Not implemented - should never be called if isAvailable() returns false
        outputs.clear();
    }

    void findNearestEntities(
        const std::vector<AgentPosition>& agent_positions,
        const std::vector<AgentPosition>& target_positions,
        std::vector<NearestEntityResult>& results,
        int search_radius
    ) override {
        // Not implemented
        results.clear();
    }

    void batchedMutate(
        const std::vector<NeuralNetwork*>& networks,
        double mutation_rate,
        double mutation_amount
    ) override {
        // Not implemented
    }
};

#endif  // USE_CUDA

#endif // CUDA_BACKEND_H
