#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernels for compute backend

// Activation function: tanh
__device__ double d_tanh(double x) {
    return tanh(x);
}

// Kernel for batched neural network forward pass
// Assumes all networks have same architecture: input_size -> hidden_size -> output_size
// With recurrent connections in hidden layer
__global__ void batchedForwardKernel(
    const double* d_inputs,      // [batch_size][input_size]
    const double* d_weights1,    // [batch_size][hidden_size][input_size]
    const double* d_biases1,     // [batch_size][hidden_size]
    const double* d_weights2,    // [batch_size][output_size][hidden_size]
    const double* d_biases2,     // [batch_size][output_size]
    const double* d_recurrent,   // [batch_size][hidden_size][hidden_size]
    const double* d_hidden_state,// [batch_size][hidden_size] (previous)
    double* d_hidden_output,     // [batch_size][hidden_size] (new hidden state)
    double* d_outputs,           // [batch_size][output_size]
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Compute hidden layer
    for (int h = 0; h < hidden_size; h++) {
        double sum = d_biases1[batch_idx * hidden_size + h];

        // Input to hidden
        for (int i = 0; i < input_size; i++) {
            sum += d_inputs[batch_idx * input_size + i] *
                   d_weights1[batch_idx * hidden_size * input_size + h * input_size + i];
        }

        // Recurrent connections
        for (int r = 0; r < hidden_size; r++) {
            sum += d_hidden_state[batch_idx * hidden_size + r] *
                   d_recurrent[batch_idx * hidden_size * hidden_size + h * hidden_size + r];
        }

        double activation = d_tanh(sum);
        d_hidden_output[batch_idx * hidden_size + h] = activation;
    }

    // Compute output layer
    for (int o = 0; o < output_size; o++) {
        double sum = d_biases2[batch_idx * output_size + o];

        // Hidden to output
        for (int h = 0; h < hidden_size; h++) {
            sum += d_hidden_output[batch_idx * hidden_size + h] *
                   d_weights2[batch_idx * output_size * hidden_size + o * hidden_size + h];
        }

        d_outputs[batch_idx * output_size + o] = d_tanh(sum);
    }
}

// Kernel for finding nearest entities
__global__ void findNearestEntitiesKernel(
    const int* d_agent_x,        // [agent_count]
    const int* d_agent_y,        // [agent_count]
    const int* d_target_x,       // [target_count]
    const int* d_target_y,       // [target_count]
    int* d_result_dx,            // [agent_count]
    int* d_result_dy,            // [agent_count]
    int* d_result_distance,      // [agent_count]
    int* d_result_found,         // [agent_count]
    int agent_count,
    int target_count,
    int search_radius
) {
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_idx >= agent_count) return;

    int agent_x = d_agent_x[agent_idx];
    int agent_y = d_agent_y[agent_idx];

    int min_distance = 999999;
    int best_dx = 0;
    int best_dy = 0;
    bool found = false;

    // Search all targets
    for (int t = 0; t < target_count; t++) {
        int target_x = d_target_x[t];
        int target_y = d_target_y[t];

        int dx = target_x - agent_x;
        int dy = target_y - agent_y;

        // Skip if outside search radius
        if (abs(dx) > search_radius || abs(dy) > search_radius) continue;

        int distance = abs(dx) + abs(dy);

        if (distance < min_distance) {
            min_distance = distance;
            best_dx = dx;
            best_dy = dy;
            found = true;
        }
    }

    d_result_dx[agent_idx] = best_dx;
    d_result_dy[agent_idx] = best_dy;
    d_result_distance[agent_idx] = min_distance;
    d_result_found[agent_idx] = found ? 1 : 0;
}

// Kernel for batched mutation
__global__ void batchedMutateKernel(
    double* d_weights,           // [total_weights]
    const double* d_random_prob, // [total_weights] random values 0-1
    const double* d_random_amount,// [total_weights] random values for mutation
    double mutation_rate,
    double mutation_amount,
    int total_weights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_weights) return;

    if (d_random_prob[idx] < mutation_rate) {
        d_weights[idx] += d_random_amount[idx] * mutation_amount;
    }
}

// Host functions to launch kernels

cudaError_t launchBatchedForward(
    const double* d_inputs,
    const double* d_weights1, const double* d_biases1,
    const double* d_weights2, const double* d_biases2,
    const double* d_recurrent, const double* d_hidden_state,
    double* d_hidden_output, double* d_outputs,
    int batch_size, int input_size, int hidden_size, int output_size
) {
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    batchedForwardKernel<<<grid, block>>>(
        d_inputs, d_weights1, d_biases1, d_weights2, d_biases2,
        d_recurrent, d_hidden_state, d_hidden_output, d_outputs,
        batch_size, input_size, hidden_size, output_size
    );

    return cudaGetLastError();
}

cudaError_t launchFindNearestEntities(
    const int* d_agent_x, const int* d_agent_y,
    const int* d_target_x, const int* d_target_y,
    int* d_result_dx, int* d_result_dy, int* d_result_distance, int* d_result_found,
    int agent_count, int target_count, int search_radius
) {
    dim3 block(256);
    dim3 grid((agent_count + block.x - 1) / block.x);

    findNearestEntitiesKernel<<<grid, block>>>(
        d_agent_x, d_agent_y, d_target_x, d_target_y,
        d_result_dx, d_result_dy, d_result_distance, d_result_found,
        agent_count, target_count, search_radius
    );

    return cudaGetLastError();
}

cudaError_t launchBatchedMutate(
    double* d_weights,
    const double* d_random_prob, const double* d_random_amount,
    double mutation_rate, double mutation_amount, int total_weights
) {
    dim3 block(256);
    dim3 grid((total_weights + block.x - 1) / block.x);

    batchedMutateKernel<<<grid, block>>>(
        d_weights, d_random_prob, d_random_amount,
        mutation_rate, mutation_amount, total_weights
    );

    return cudaGetLastError();
}