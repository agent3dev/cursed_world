#include "../include/ComputeBackend.h"
#include "../include/CPUBackend.h"
#include "../include/CUDABackend.h"
#include <iostream>
#include <algorithm>
#include <cctype>

std::unique_ptr<ComputeBackend> createComputeBackend(
    BackendType type,
    bool fallback_to_cpu
) {
    std::unique_ptr<ComputeBackend> backend;

    // Handle AUTO mode
    if (type == BackendType::AUTO) {
        // Try CUDA first, then fall back to CPU
        backend = std::make_unique<CUDABackend>();
        if (backend->initialize() && backend->isAvailable()) {
            std::cout << "[ComputeBackend] AUTO: Selected CUDA backend\n";
            return backend;
        }

        // CUDA not available, use CPU
        std::cout << "[ComputeBackend] AUTO: CUDA unavailable, using CPU backend\n";
        backend = std::make_unique<CPUBackend>();
        if (backend->initialize()) {
            return backend;
        }

        std::cerr << "[ComputeBackend] ERROR: Failed to initialize any backend!\n";
        return nullptr;
    }

    // Handle explicit backend selection
    if (type == BackendType::CUDA) {
        backend = std::make_unique<CUDABackend>();

        if (!backend->isAvailable()) {
            std::cerr << "[ComputeBackend] CUDA backend not available on this system\n";

            if (fallback_to_cpu) {
                std::cout << "[ComputeBackend] Falling back to CPU backend\n";
                backend = std::make_unique<CPUBackend>();
            } else {
                std::cerr << "[ComputeBackend] Fallback disabled, no backend available\n";
                return nullptr;
            }
        }

        if (!backend->initialize()) {
            std::cerr << "[ComputeBackend] Failed to initialize CUDA backend\n";

            if (fallback_to_cpu) {
                std::cout << "[ComputeBackend] Falling back to CPU backend\n";
                backend = std::make_unique<CPUBackend>();
                if (!backend->initialize()) {
                    std::cerr << "[ComputeBackend] Failed to initialize CPU backend\n";
                    return nullptr;
                }
            } else {
                return nullptr;
            }
        }

        return backend;
    }

    // Handle CPU backend (default)
    if (type == BackendType::CPU) {
        backend = std::make_unique<CPUBackend>();
        if (!backend->initialize()) {
            std::cerr << "[ComputeBackend] Failed to initialize CPU backend\n";
            return nullptr;
        }
        return backend;
    }

    std::cerr << "[ComputeBackend] Unknown backend type\n";
    return nullptr;
}

const char* backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::CPU:  return "cpu";
        case BackendType::CUDA: return "cuda";
        case BackendType::AUTO: return "auto";
        default: return "unknown";
    }
}

BackendType stringToBackendType(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower == "cpu") {
        return BackendType::CPU;
    } else if (lower == "cuda" || lower == "gpu") {
        return BackendType::CUDA;
    } else if (lower == "auto") {
        return BackendType::AUTO;
    }

    // Default to CPU if unrecognized
    return BackendType::CPU;
}
