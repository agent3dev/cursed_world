#include "../include/CUDABackend.h"
#include <iostream>

// Only compile CUDA implementation if USE_CUDA is defined
#ifdef USE_CUDA

#include "../include/NeuralNetwork.h"
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

// Forward declarations for CUDA kernels
extern cudaError_t launchBatchedForward(
    const double* d_inputs,
    const double* d_weights1, const double* d_biases1,
    const double* d_weights2, const double* d_biases2,
    const double* d_recurrent, const double* d_hidden_state,
    double* d_hidden_output, double* d_outputs,
    int batch_size, int input_size, int hidden_size, int output_size
);

extern cudaError_t launchFindNearestEntities(
    const int* d_agent_x, const int* d_agent_y,
    const int* d_target_x, const int* d_target_y,
    int* d_result_dx, int* d_result_dy, int* d_result_distance, int* d_result_found,
    int agent_count, int target_count, int search_radius
);

extern cudaError_t launchBatchedMutate(
    double* d_weights,
    const double* d_random_prob, const double* d_random_amount,
    double mutation_rate, double mutation_amount, int total_weights
);

CUDABackend::CUDABackend()
    : initialized_(false), device_id_(0),
      d_inputs_(nullptr), d_outputs_(nullptr),
      d_weights1_(nullptr), d_biases1_(nullptr),
      d_weights2_(nullptr), d_biases2_(nullptr),
      d_recurrent_(nullptr), d_hidden_state_(nullptr), d_hidden_output_(nullptr),
      d_agent_x_(nullptr), d_agent_y_(nullptr),
      d_target_x_(nullptr), d_target_y_(nullptr),
      d_result_dx_(nullptr), d_result_dy_(nullptr),
      d_result_distance_(nullptr), d_result_found_(nullptr),
      d_weights_all_(nullptr), d_random_prob_(nullptr), d_random_amount_(nullptr),
      allocated_size_(0), max_batch_size_(0),
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
    cudaError_t error;

    // Allocate buffers for neural network operations (assuming 9-16-9 architecture)
    const size_t INPUT_SIZE = 9;
    const size_t HIDDEN_SIZE = 16;
    const size_t OUTPUT_SIZE = 9;
    const size_t MAX_BATCH = 1024;  // Maximum batch size we can handle
    const size_t MAX_WEIGHTS = MAX_BATCH * (HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE + HIDDEN_SIZE * HIDDEN_SIZE);

    // Neural network buffers
    if (cudaMalloc(&d_inputs_, MAX_BATCH * INPUT_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_outputs_, MAX_BATCH * OUTPUT_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_weights1_, MAX_BATCH * HIDDEN_SIZE * INPUT_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_biases1_, MAX_BATCH * HIDDEN_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_weights2_, MAX_BATCH * OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_biases2_, MAX_BATCH * OUTPUT_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_recurrent_, MAX_BATCH * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_hidden_state_, MAX_BATCH * HIDDEN_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_hidden_output_, MAX_BATCH * HIDDEN_SIZE * sizeof(double)) != cudaSuccess) goto cleanup;

    // Distance search buffers
    if (cudaMalloc(&d_agent_x_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_agent_y_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_target_x_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_target_y_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_result_dx_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_result_dy_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_result_distance_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_result_found_, MAX_BATCH * sizeof(int)) != cudaSuccess) goto cleanup;

    // Mutation buffers
    if (cudaMalloc(&d_weights_all_, MAX_WEIGHTS * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_random_prob_, MAX_WEIGHTS * sizeof(double)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_random_amount_, MAX_WEIGHTS * sizeof(double)) != cudaSuccess) goto cleanup;

    allocated_size_ = size;
    max_batch_size_ = MAX_BATCH;
    return true;

cleanup:
    freeGPUMemory();
    std::cerr << "[CUDABackend] GPU memory allocation failed\n";
    return false;
}

void CUDABackend::freeGPUMemory() {
    if (d_inputs_) cudaFree(d_inputs_);
    if (d_outputs_) cudaFree(d_outputs_);
    if (d_weights1_) cudaFree(d_weights1_);
    if (d_biases1_) cudaFree(d_biases1_);
    if (d_weights2_) cudaFree(d_weights2_);
    if (d_biases2_) cudaFree(d_biases2_);
    if (d_recurrent_) cudaFree(d_recurrent_);
    if (d_hidden_state_) cudaFree(d_hidden_state_);
    if (d_hidden_output_) cudaFree(d_hidden_output_);

    if (d_agent_x_) cudaFree(d_agent_x_);
    if (d_agent_y_) cudaFree(d_agent_y_);
    if (d_target_x_) cudaFree(d_target_x_);
    if (d_target_y_) cudaFree(d_target_y_);
    if (d_result_dx_) cudaFree(d_result_dx_);
    if (d_result_dy_) cudaFree(d_result_dy_);
    if (d_result_distance_) cudaFree(d_result_distance_);
    if (d_result_found_) cudaFree(d_result_found_);

    if (d_weights_all_) cudaFree(d_weights_all_);
    if (d_random_prob_) cudaFree(d_random_prob_);
    if (d_random_amount_) cudaFree(d_random_amount_);

    d_inputs_ = nullptr;
    d_outputs_ = nullptr;
    d_weights1_ = nullptr;
    d_biases1_ = nullptr;
    d_weights2_ = nullptr;
    d_biases2_ = nullptr;
    d_recurrent_ = nullptr;
    d_hidden_state_ = nullptr;
    d_hidden_output_ = nullptr;

    d_agent_x_ = nullptr;
    d_agent_y_ = nullptr;
    d_target_x_ = nullptr;
    d_target_y_ = nullptr;
    d_result_dx_ = nullptr;
    d_result_dy_ = nullptr;
    d_result_distance_ = nullptr;
    d_result_found_ = nullptr;

    d_weights_all_ = nullptr;
    d_random_prob_ = nullptr;
    d_random_amount_ = nullptr;

    allocated_size_ = 0;
    max_batch_size_ = 0;
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

    size_t batch_size = inputs.size();
    outputs.resize(batch_size);

    if (batch_size == 0) return;

    // Network architecture constants
    const int INPUT_SIZE = 9;
    const int HIDDEN_SIZE = 16;
    const int OUTPUT_SIZE = 9;

    // Ensure GPU memory is allocated
    if (!allocateGPUMemory(batch_size)) {
        std::cerr << "[CUDABackend] GPU memory allocation failed, falling back to CPU\n";
        goto cpu_fallback;
    }

    // Check batch size fits in allocated memory
    if (batch_size > max_batch_size_) {
        std::cerr << "[CUDABackend] Batch size " << batch_size << " exceeds max " << max_batch_size_ << ", falling back to CPU\n";
        goto cpu_fallback;
    }

    // Marshal data to host buffers first
    {
        std::vector<double> h_inputs(batch_size * INPUT_SIZE);
        std::vector<double> h_weights1(batch_size * HIDDEN_SIZE * INPUT_SIZE);
        std::vector<double> h_biases1(batch_size * HIDDEN_SIZE);
        std::vector<double> h_weights2(batch_size * OUTPUT_SIZE * HIDDEN_SIZE);
        std::vector<double> h_biases2(batch_size * OUTPUT_SIZE);
        std::vector<double> h_recurrent(batch_size * HIDDEN_SIZE * HIDDEN_SIZE);
        std::vector<double> h_hidden_state(batch_size * HIDDEN_SIZE, 0.0);
        std::vector<double> h_hidden_output(batch_size * HIDDEN_SIZE);
        std::vector<double> h_outputs(batch_size * OUTPUT_SIZE);

        // Copy inputs and network weights
        for (size_t i = 0; i < batch_size; i++) {
            // Copy inputs
            for (int j = 0; j < INPUT_SIZE && j < (int)inputs[i].size(); j++) {
                h_inputs[i * INPUT_SIZE + j] = inputs[i][j];
            }

            // Copy network weights and biases
            if (i < networks.size() && networks[i]) {
                const auto& net = networks[i];
                const auto& w1 = net->getWeights1();
                const auto& b1 = net->getBiases1();
                const auto& w2 = net->getWeights2();
                const auto& b2 = net->getBiases2();
                const auto& rec = net->getRecurrentWeights();
                const auto& hidden = net->getHiddenState();

                // Flatten weights1: [HIDDEN_SIZE][INPUT_SIZE]
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    for (int in = 0; in < INPUT_SIZE; in++) {
                        h_weights1[i * HIDDEN_SIZE * INPUT_SIZE + h * INPUT_SIZE + in] = w1[h][in];
                    }
                }

                // Copy biases1
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    h_biases1[i * HIDDEN_SIZE + h] = b1[h];
                }

                // Flatten weights2: [OUTPUT_SIZE][HIDDEN_SIZE]
                for (int o = 0; o < OUTPUT_SIZE; o++) {
                    for (int h = 0; h < HIDDEN_SIZE; h++) {
                        h_weights2[i * OUTPUT_SIZE * HIDDEN_SIZE + o * HIDDEN_SIZE + h] = w2[o][h];
                    }
                }

                // Copy biases2
                for (int o = 0; o < OUTPUT_SIZE; o++) {
                    h_biases2[i * OUTPUT_SIZE + o] = b2[o];
                }

                // Flatten recurrent: [HIDDEN_SIZE][HIDDEN_SIZE]
                for (int h1 = 0; h1 < HIDDEN_SIZE; h1++) {
                    for (int h2 = 0; h2 < HIDDEN_SIZE; h2++) {
                        h_recurrent[i * HIDDEN_SIZE * HIDDEN_SIZE + h1 * HIDDEN_SIZE + h2] = rec[h1][h2];
                    }
                }

                // Copy hidden state
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    h_hidden_state[i * HIDDEN_SIZE + h] = hidden[h];
                }
            }
        }

        // Transfer to GPU
        cudaError_t err;
        err = cudaMemcpy(d_inputs_, h_inputs.data(), batch_size * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_weights1_, h_weights1.data(), batch_size * HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_biases1_, h_biases1.data(), batch_size * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_weights2_, h_weights2.data(), batch_size * OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_biases2_, h_biases2.data(), batch_size * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_recurrent_, h_recurrent.data(), batch_size * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_hidden_state_, h_hidden_state.data(), batch_size * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        // Launch kernel
        err = launchBatchedForward(
            d_inputs_, d_weights1_, d_biases1_, d_weights2_, d_biases2_,
            d_recurrent_, d_hidden_state_, d_hidden_output_, d_outputs_,
            batch_size, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
        );
        if (err != cudaSuccess) goto gpu_error;

        // Wait for completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto gpu_error;

        // Copy results back
        err = cudaMemcpy(h_outputs.data(), d_outputs_, batch_size * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(h_hidden_output.data(), d_hidden_output_, batch_size * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto gpu_error;

        // Unpack results
        for (size_t i = 0; i < batch_size; i++) {
            outputs[i].resize(OUTPUT_SIZE);
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                outputs[i][j] = h_outputs[i * OUTPUT_SIZE + j];
            }

            // Update hidden state in network
            if (i < networks.size() && networks[i]) {
                std::vector<double> new_hidden(HIDDEN_SIZE);
                for (int h = 0; h < HIDDEN_SIZE; h++) {
                    new_hidden[h] = h_hidden_output[i * HIDDEN_SIZE + h];
                }
                networks[i]->setHiddenState(new_hidden);
            }
        }
    }

    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time_ms_ += duration.count() / 1000.0;
        operation_count_++;
    }
    return;

gpu_error:
    std::cerr << "[CUDABackend] GPU error in batchedForward: " << cudaGetErrorString(cudaGetLastError()) << ", falling back to CPU\n";

cpu_fallback:
    // CPU fallback implementation
    for (size_t i = 0; i < batch_size; i++) {
        if (i < networks.size() && networks[i]) {
            outputs[i] = networks[i]->forward(inputs[i]);
        } else {
            outputs[i] = std::vector<double>(OUTPUT_SIZE, 0.0);
        }
    }

    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time_ms_ += duration.count() / 1000.0;
        operation_count_++;
    }
}

void CUDABackend::findNearestEntities(
    const std::vector<AgentPosition>& agent_positions,
    const std::vector<AgentPosition>& target_positions,
    std::vector<NearestEntityResult>& results,
    int search_radius
) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t agent_count = agent_positions.size();
    size_t target_count = target_positions.size();
    results.resize(agent_count);

    if (agent_count == 0) return;

    // Ensure GPU memory is allocated
    if (!allocateGPUMemory(agent_count)) {
        std::cerr << "[CUDABackend] GPU memory allocation failed, falling back to CPU\n";
        goto cpu_fallback;
    }

    // Check counts fit in allocated memory
    if (agent_count > max_batch_size_ || target_count > max_batch_size_) {
        std::cerr << "[CUDABackend] Agent/target count exceeds max " << max_batch_size_ << ", falling back to CPU\n";
        goto cpu_fallback;
    }

    // Marshal data to host buffers
    {
        std::vector<int> h_agent_x(agent_count);
        std::vector<int> h_agent_y(agent_count);
        std::vector<int> h_target_x(target_count);
        std::vector<int> h_target_y(target_count);
        std::vector<int> h_result_dx(agent_count);
        std::vector<int> h_result_dy(agent_count);
        std::vector<int> h_result_distance(agent_count);
        std::vector<int> h_result_found(agent_count);

        // Copy agent positions
        for (size_t i = 0; i < agent_count; i++) {
            h_agent_x[i] = agent_positions[i].x;
            h_agent_y[i] = agent_positions[i].y;
        }

        // Copy target positions
        for (size_t i = 0; i < target_count; i++) {
            h_target_x[i] = target_positions[i].x;
            h_target_y[i] = target_positions[i].y;
        }

        // Transfer to GPU
        cudaError_t err;
        err = cudaMemcpy(d_agent_x_, h_agent_x.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_agent_y_, h_agent_y.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_target_x_, h_target_x.data(), target_count * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(d_target_y_, h_target_y.data(), target_count * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto gpu_error;

        // Launch kernel
        err = launchFindNearestEntities(
            d_agent_x_, d_agent_y_, d_target_x_, d_target_y_,
            d_result_dx_, d_result_dy_, d_result_distance_, d_result_found_,
            agent_count, target_count, search_radius
        );
        if (err != cudaSuccess) goto gpu_error;

        // Wait for completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto gpu_error;

        // Copy results back
        err = cudaMemcpy(h_result_dx.data(), d_result_dx_, agent_count * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(h_result_dy.data(), d_result_dy_, agent_count * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(h_result_distance.data(), d_result_distance_, agent_count * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto gpu_error;

        err = cudaMemcpy(h_result_found.data(), d_result_found_, agent_count * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto gpu_error;

        // Unpack results
        for (size_t i = 0; i < agent_count; i++) {
            results[i].dx = h_result_dx[i];
            results[i].dy = h_result_dy[i];
            results[i].distance = h_result_distance[i];
            results[i].found = (h_result_found[i] != 0);
        }
    }

    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time_ms_ += duration.count() / 1000.0;
        operation_count_++;
    }
    return;

gpu_error:
    std::cerr << "[CUDABackend] GPU error in findNearestEntities: " << cudaGetErrorString(cudaGetLastError()) << ", falling back to CPU\n";

cpu_fallback:
    // CPU fallback implementation
    for (size_t i = 0; i < agent_count; i++) {
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

    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time_ms_ += duration.count() / 1000.0;
        operation_count_++;
    }
}

void CUDABackend::batchedMutate(
    const std::vector<NeuralNetwork*>& networks,
    double mutation_rate,
    double mutation_amount
) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t batch_size = networks.size();
    if (batch_size == 0) return;

    // Note: Mutation kernel is implemented but requires cuRAND for GPU-side random generation
    // For now, we use CPU fallback as it's simpler and mutation is not the bottleneck
    // TODO: Implement cuRAND-based GPU mutation in Phase 4 optimization

    // Skip GPU implementation for now (unreachable code wrapped to avoid goto errors)
    if (false) {
        // Network architecture constants
        const int INPUT_SIZE = 9;
        const int HIDDEN_SIZE = 16;
        const int OUTPUT_SIZE = 9;
        const size_t weights_per_network =
            HIDDEN_SIZE * INPUT_SIZE +  // weights1
            HIDDEN_SIZE +               // biases1
            OUTPUT_SIZE * HIDDEN_SIZE + // weights2
            OUTPUT_SIZE +               // biases2
            HIDDEN_SIZE * HIDDEN_SIZE;  // recurrent

        size_t total_weights = batch_size * weights_per_network;
        (void)total_weights; // Suppress unused warning

        // For mutation we'd need:
        // 1. Transfer all weights to GPU
        // 2. Generate random numbers (requires cuRAND or pre-generated on CPU)
        // 3. Apply mutations in parallel
        // 4. Transfer weights back
        // 5. Update networks
        //
        // This adds significant complexity and CPU<->GPU transfer overhead
        // Mutation is typically much faster than forward passes, so GPU acceleration
        // provides minimal benefit. Defer to Phase 4 optimization.
    }

cpu_fallback:
    // CPU fallback implementation
    for (NeuralNetwork* network : networks) {
        if (network) {
            network->mutate(mutation_rate, mutation_amount);
        }
    }

    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time_ms_ += duration.count() / 1000.0;
        operation_count_++;
    }
}

#endif  // USE_CUDA
