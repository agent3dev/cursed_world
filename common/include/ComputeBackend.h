#ifndef COMPUTE_BACKEND_H
#define COMPUTE_BACKEND_H

#include <vector>
#include <string>
#include <memory>

/**
 * Abstract interface for compute backends (CPU, CUDA, etc.)
 *
 * This allows the simulation to switch between CPU and GPU computation
 * at compile-time (via preprocessor flags) or runtime (via configuration).
 */

// Forward declarations
class NeuralNetwork;

/**
 * Backend type enumeration
 */
enum class BackendType {
    CPU,    // Always available, no dependencies
    CUDA,   // GPU acceleration (requires CUDA toolkit)
    AUTO    // Automatically select best available backend
};

/**
 * Result structure for nearest entity searches
 */
struct NearestEntityResult {
    int dx, dy;        // Direction vector to target
    int distance;      // Manhattan distance
    bool found;        // Whether any entity was found

    NearestEntityResult() : dx(0), dy(0), distance(999999), found(false) {}
};

/**
 * Agent position structure for batch operations
 */
struct AgentPosition {
    int x, y;
    int id;  // Agent ID for tracking
};

/**
 * Abstract compute backend interface
 *
 * All compute-intensive operations should be implemented through this interface
 * to allow transparent switching between CPU and GPU implementations.
 */
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    /**
     * Get the backend type
     */
    virtual BackendType getType() const = 0;

    /**
     * Get backend name for display
     */
    virtual std::string getName() const = 0;

    /**
     * Initialize the backend (allocate resources, check GPU availability, etc.)
     * Returns true if initialization succeeded
     */
    virtual bool initialize() = 0;

    /**
     * Cleanup backend resources
     */
    virtual void cleanup() = 0;

    /**
     * Check if backend is available on this system
     */
    virtual bool isAvailable() const = 0;

    /**
     * Batched neural network forward pass
     *
     * @param inputs - Vector of input vectors (one per agent)
     * @param networks - Vector of neural network pointers
     * @param outputs - Vector to store output vectors (one per agent)
     */
    virtual void batchedForward(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<NeuralNetwork*>& networks,
        std::vector<std::vector<double>>& outputs
    ) = 0;

    /**
     * Find nearest entities of a specific type
     *
     * @param agent_positions - Positions of agents searching
     * @param target_positions - Positions of potential targets
     * @param results - Output: nearest entity for each agent
     * @param search_radius - Maximum search distance
     */
    virtual void findNearestEntities(
        const std::vector<AgentPosition>& agent_positions,
        const std::vector<AgentPosition>& target_positions,
        std::vector<NearestEntityResult>& results,
        int search_radius
    ) = 0;

    /**
     * Batched neural network weight mutation
     *
     * @param networks - Vector of neural networks to mutate
     * @param mutation_rate - Probability of mutating each weight (0.0-1.0)
     * @param mutation_amount - Range of mutation (-amount to +amount)
     */
    virtual void batchedMutate(
        const std::vector<NeuralNetwork*>& networks,
        double mutation_rate,
        double mutation_amount
    ) = 0;

    /**
     * Get performance statistics (optional, for benchmarking)
     */
    virtual void getStats(double& total_time_ms, int& operation_count) const {
        total_time_ms = 0.0;
        operation_count = 0;
    }

    /**
     * Reset performance statistics
     */
    virtual void resetStats() {}
};

/**
 * Factory function to create compute backend
 *
 * @param type - Backend type to create
 * @param fallback_to_cpu - If true, fall back to CPU if requested backend unavailable
 * @return Pointer to created backend (caller owns memory)
 */
std::unique_ptr<ComputeBackend> createComputeBackend(
    BackendType type = BackendType::AUTO,
    bool fallback_to_cpu = true
);

/**
 * Convert backend type to string
 */
const char* backendTypeToString(BackendType type);

/**
 * Parse backend type from string
 * Returns BackendType::CPU if string not recognized
 */
BackendType stringToBackendType(const std::string& str);

#endif // COMPUTE_BACKEND_H
