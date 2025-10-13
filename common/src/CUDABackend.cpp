#include "../include/CUDABackend.h"
#include <iostream>

// Only compile CUDA implementation if USE_CUDA is defined
#ifdef USE_CUDA

#include "../include/NeuralNetwork.h"
#include <cstring>

CUDABackend::CUDABackend()
    : initialized_(false), device_id_(0), d_weights_(nullptr),
      d_inputs_(nullptr), d_outputs_(nullptr), allocated_size_(0),
      total_time_ms_(0.0), operation_count_(0) {
    memset(&device_properties_, 0, sizeof(device_properties_));
}

CUDABackend::~CUDABackend() {
    cleanup();
}

bool CUDABackend::checkCudaDevice() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "[CUDABackend] No CUDA-capable devices found\n";
        return false;
    }

    // Use device 0 by default
    device_id_ = 0;
    error = cudaGetDeviceProperties(&device_properties_, device_id_);

    if (error != cudaSuccess) {
        std::cerr << "[CUDABackend] Failed to get device properties\n";
        return false;
    }

    std::cout << "[CUDABackend] Found CUDA device: " << device_properties_.name << "\n";
    std::cout << "[CUDABackend] Compute capability: "
              << device_properties_.major << "." << device_properties_.minor << "\n";
    std::cout << "[CUDABackend] Total memory: "
              << (device_properties_.totalGlobalMem / (1024 * 1024)) << " MB\n";

    // Check minimum compute capability (6.0)
    if (device_properties_.major < 6) {
        std::cerr << "[CUDABackend] WARNING: Compute capability < 6.0 may have limited support\n";
    }

    return true;
}

bool CUDABackend::allocateGPUMemory(size_t size) {
    if (allocated_size_ >= size) {
        return true;  // Already have enough memory
    }

    // Free old memory if any
    freeGPUMemory();

    // Allocate new memory
    // TODO: Implement proper GPU memory allocation when kernels are ready
    allocated_size_ = size;
    return true;
}

void CUDABackend::freeGPUMemory() {
    if (d_weights_) {
        cudaFree(d_weights_);
        d_weights_ = nullptr;
    }
    if (d_inputs_) {
        cudaFree(d_inputs_);
        d_inputs_ = nullptr;
    }
    if (d_outputs_) {
        cudaFree(d_outputs_);
        d_outputs_ = nullptr;
    }
    allocated_size_ = 0;
}

std::string CUDABackend::getName() const {
    if (initialized_) {
        return std::string("CUDA (") + device_properties_.name + ")";
    }
    return "CUDA";
}

bool CUDABackend::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "[CUDABackend] Initializing CUDA compute backend...\n";

    if (!checkCudaDevice()) {
        return false;
    }

    cudaError_t error = cudaSetDevice(device_id_);
    if (error != cudaSuccess) {
        std::cerr << "[CUDABackend] Failed to set CUDA device\n";
        return false;
    }

    initialized_ = true;
    resetStats();
    std::cout << "[CUDABackend] Initialization successful\n";
    return true;
}

void CUDABackend::cleanup() {
    if (!initialized_) {
        return;
    }

    std::cout << "[CUDABackend] Cleaning up CUDA backend...\n";
    freeGPUMemory();

    cudaDeviceReset();
    initialized_ = false;
}

bool CUDABackend::isAvailable() const {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

void CUDABackend::batchedForward(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<NeuralNetwork*>& networks,
    std::vector<std::vector<double>>& outputs
) {
    auto start = std::chrono::high_resolution_clock::now();

    // TODO: Implement CUDA kernel for batched forward pass
    // For now, fall back to CPU implementation
    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        if (i < networks.size() && networks[i]) {
            outputs[i] = networks[i]->forward(inputs[i]);
        } else {
            outputs[i] = std::vector<double>(9, 0.0);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_ms_ += duration.count() / 1000.0;
    operation_count_++;
}

void CUDABackend::findNearestEntities(
    const std::vector<AgentPosition>& agent_positions,
    const std::vector<AgentPosition>& target_positions,
    std::vector<NearestEntityResult>& results,
    int search_radius
) {
    auto start = std::chrono::high_resolution_clock::now();

    // TODO: Implement CUDA kernel for parallel nearest entity search
    // For now, fall back to CPU implementation
    results.resize(agent_positions.size());
    for (size_t i = 0; i < agent_positions.size(); i++) {
        NearestEntityResult& result = results[i];
        result.found = false;
        result.distance = 999999;

        const AgentPosition& agent = agent_positions[i];

        for (const AgentPosition& target : target_positions) {
            int dx = target.x - agent.x;
            int dy = target.y - agent.y;

            if (std::abs(dx) > search_radius || std::abs(dy) > search_radius) {
                continue;
            }

            int distance = std::abs(dx) + std::abs(dy);

            if (distance < result.distance) {
                result.dx = dx;
                result.dy = dy;
                result.distance = distance;
                result.found = true;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time_ms_ += duration.count() / 1000.0;
    operation_count_++;
}

void CUDABackend::batchedMutate(
    const std::vector<NeuralNetwork*>& networks,
    double mutation_rate,
    double mutation_amount
) {
    auto start = std::chrono::high_resolution_clock::now();

    // TODO: Implement CUDA kernel for parallel mutation
    // For now, fall back to CPU implementation
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

#endif  // USE_CUDA
