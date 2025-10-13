# CUDA Toggleable Architecture Design

**Date**: 2025-10-11
**Goal**: Make CUDA functionality optional and switchable at compile-time and runtime

---

## Design Philosophy

The system should support:
1. **Compile without CUDA** - Works on machines without NVIDIA GPU/CUDA toolkit
2. **Compile with CUDA** - Enables GPU acceleration when available
3. **Runtime toggle** - Switch between CPU/GPU at runtime for benchmarking
4. **Automatic fallback** - Gracefully fall back to CPU if GPU unavailable

---

## Architecture Overview

```
┌─────────────────────────────────────┐
│   Application Layer                 │
│   (Evolution Simulation)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Compute Interface (Abstract)      │
│   - computeNeuralNetworks()         │
│   - findNearestEntities()           │
│   - mutateWeights()                 │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│ CPU Backend │  │ CUDA Backend │
│ (Always)    │  │ (Optional)   │
└─────────────┘  └──────────────┘
```

---

## Implementation Strategy

### 1. Preprocessor Flags

**In Makefile**:
```makefile
# User can set ENABLE_CUDA=1 to enable GPU support
ENABLE_CUDA ?= 0

ifeq ($(ENABLE_CUDA), 1)
    NVCC = nvcc
    CUDA_FLAGS = -std=c++14 -arch=sm_61
    CXXFLAGS += -DUSE_CUDA
    LDFLAGS += -lcudart -L/usr/local/cuda/lib64
else
    # CPU-only build
endif
```

**Usage**:
```bash
# CPU-only build (default)
make

# GPU-enabled build
make ENABLE_CUDA=1
```

---

### 2. Abstraction Layer: ComputeBackend

**File**: `common/include/ComputeBackend.h`

```cpp
#ifndef COMPUTE_BACKEND_H
#define COMPUTE_BACKEND_H

#include <vector>
#include <string>

// Enumeration for backend types
enum class BackendType {
    CPU,
    CUDA,
    AUTO  // Automatically choose best available
};

// Structure for entity positions (used in distance calculations)
struct EntityPositions {
    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> types;  // RODENT, CAT, etc.
    int count;
};

// Structure for nearest entity results
struct NearestEntityBatch {
    std::vector<int> dx;        // Direction X for each agent
    std::vector<int> dy;        // Direction Y for each agent
    std::vector<int> distance;  // Distance for each agent
    std::vector<bool> found;    // Whether entity was found
};

// Abstract interface for compute backends
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Get backend type
    virtual BackendType getType() const = 0;
    virtual std::string getName() const = 0;

    // Check if backend is available
    virtual bool isAvailable() const = 0;

    // Initialize backend
    virtual bool initialize() = 0;

    // Neural network operations
    virtual void batchedForward(
        const std::vector<std::vector<double>>& inputs,    // [N_agents][input_size]
        const std::vector<std::vector<double>>& weights,   // [N_agents][weight_count]
        std::vector<std::vector<double>>& outputs          // [N_agents][output_size]
    ) = 0;

    // Distance calculation operations
    virtual void findNearestEntities(
        const EntityPositions& agents,
        const EntityPositions& targets,
        NearestEntityBatch& results,
        int search_radius
    ) = 0;

    // Mutation operations
    virtual void batchedMutate(
        std::vector<std::vector<double>>& weights,  // [N_agents][weight_count]
        double mutation_rate,
        double mutation_amount
    ) = 0;

    // Benchmark utilities
    virtual double getLastOperationTimeMs() const { return 0.0; }
};

// Factory function to create appropriate backend
ComputeBackend* createComputeBackend(BackendType type = BackendType::AUTO);

// Global backend instance (can be switched at runtime)
extern ComputeBackend* g_compute_backend;

// Runtime toggle function
bool setComputeBackend(BackendType type);

#endif // COMPUTE_BACKEND_H
```

---

### 3. CPU Backend (Always Available)

**File**: `common/include/CPUBackend.h`

```cpp
#ifndef CPU_BACKEND_H
#define CPU_BACKEND_H

#include "ComputeBackend.h"
#include <chrono>

class CPUBackend : public ComputeBackend {
private:
    double last_operation_time_ms;
    std::chrono::high_resolution_clock::time_point start_time;

    void startTimer();
    void stopTimer();

public:
    CPUBackend();
    ~CPUBackend() override = default;

    BackendType getType() const override { return BackendType::CPU; }
    std::string getName() const override { return "CPU (Standard)"; }
    bool isAvailable() const override { return true; }  // Always available
    bool initialize() override { return true; }

    void batchedForward(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& weights,
        std::vector<std::vector<double>>& outputs
    ) override;

    void findNearestEntities(
        const EntityPositions& agents,
        const EntityPositions& targets,
        NearestEntityBatch& results,
        int search_radius
    ) override;

    void batchedMutate(
        std::vector<std::vector<double>>& weights,
        double mutation_rate,
        double mutation_amount
    ) override;

    double getLastOperationTimeMs() const override { return last_operation_time_ms; }
};

#endif // CPU_BACKEND_H
```

---

### 4. CUDA Backend (Conditional)

**File**: `common/include/CUDABackend.h`

```cpp
#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "ComputeBackend.h"

#ifdef USE_CUDA

class CUDABackend : public ComputeBackend {
private:
    bool initialized;
    int device_id;
    size_t available_memory;
    double last_operation_time_ms;

    // GPU memory buffers (kept between calls for efficiency)
    void* d_inputs;
    void* d_weights;
    void* d_outputs;
    size_t allocated_size;

    void allocateGPUMemory(size_t size);
    void freeGPUMemory();

public:
    CUDABackend();
    ~CUDABackend() override;

    BackendType getType() const override { return BackendType::CUDA; }
    std::string getName() const override;
    bool isAvailable() const override;
    bool initialize() override;

    void batchedForward(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& weights,
        std::vector<std::vector<double>>& outputs
    ) override;

    void findNearestEntities(
        const EntityPositions& agents,
        const EntityPositions& targets,
        NearestEntityBatch& results,
        int search_radius
    ) override;

    void batchedMutate(
        std::vector<std::vector<double>>& weights,
        double mutation_rate,
        double mutation_amount
    ) override;

    double getLastOperationTimeMs() const override { return last_operation_time_ms; }

    // CUDA-specific utilities
    int getDeviceId() const { return device_id; }
    size_t getAvailableMemory() const { return available_memory; }
};

#else // !USE_CUDA

// Dummy class when CUDA is not available
class CUDABackend : public ComputeBackend {
public:
    BackendType getType() const override { return BackendType::CUDA; }
    std::string getName() const override { return "CUDA (Not Available)"; }
    bool isAvailable() const override { return false; }
    bool initialize() override { return false; }

    void batchedForward(...) override { /* No-op */ }
    void findNearestEntities(...) override { /* No-op */ }
    void batchedMutate(...) override { /* No-op */ }
};

#endif // USE_CUDA

#endif // CUDA_BACKEND_H
```

---

### 5. Runtime Configuration

**File**: `common/include/ComputeConfig.h`

```cpp
#ifndef COMPUTE_CONFIG_H
#define COMPUTE_CONFIG_H

#include <string>

// Configuration structure for compute backend
struct ComputeConfig {
    BackendType backend_type;
    bool auto_select;           // Automatically choose best available
    bool enable_benchmarking;   // Track operation times
    bool verbose;               // Print backend info

    // CUDA-specific settings
    int cuda_device_id;         // Which GPU to use (-1 = auto)
    size_t cuda_memory_limit;   // Max GPU memory to use (0 = no limit)

    // Fallback behavior
    bool fallback_to_cpu;       // Fall back to CPU if GPU unavailable

    ComputeConfig()
        : backend_type(BackendType::AUTO)
        , auto_select(true)
        , enable_benchmarking(true)
        , verbose(false)
        , cuda_device_id(-1)
        , cuda_memory_limit(0)
        , fallback_to_cpu(true)
    {}

    // Load from config file
    static ComputeConfig loadFromFile(const std::string& filename);

    // Save to config file
    void saveToFile(const std::string& filename) const;
};

#endif // COMPUTE_CONFIG_H
```

**File**: `compute_config.yaml` (user-editable)

```yaml
# Compute Backend Configuration
# Options: CPU, CUDA, AUTO

backend: AUTO  # AUTO will choose best available

# CUDA settings (only used if backend=CUDA or AUTO selects CUDA)
cuda:
  device_id: -1         # -1 = auto-select best GPU
  memory_limit: 0       # 0 = no limit (MB)
  fallback_to_cpu: true # Fall back to CPU if CUDA unavailable

# Performance
benchmarking: true      # Track operation times
verbose: false          # Print backend info on startup

# Thresholds for AUTO mode
auto_select:
  min_population_for_gpu: 100  # Only use GPU if population > 100
  prefer_gpu: true             # Prefer GPU when available
```

---

### 6. Integration with Existing Code

**Modify**: `games/evolution/src/PopulationManager.cpp`

```cpp
#include "../../common/include/ComputeBackend.h"

void PopulationManager::update(TerminalMatrix& matrix) {
    BENCHMARK_SCOPE("PopulationManager::update");

    // Check if we should use batch processing
    if (g_compute_backend && population.size() >= 50) {
        // Use backend for batch operations
        updateWithBackend(matrix);
    } else {
        // Use sequential CPU processing for small populations
        updateSequential(matrix);
    }
}

void PopulationManager::updateWithBackend(TerminalMatrix& matrix) {
    // Prepare data for backend
    EntityPositions agents;
    // ... populate agents ...

    // Distance calculations (CPU or GPU)
    NearestEntityBatch nearest_cats;
    g_compute_backend->findNearestEntities(
        agents, cats_positions, nearest_cats, 20
    );

    // Neural network forward passes (CPU or GPU)
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    // ... prepare inputs ...

    g_compute_backend->batchedForward(inputs, weights, outputs);

    // ... use results ...
}
```

---

### 7. Command-Line Interface

**Usage**:

```bash
# Run with default backend (AUTO)
./evolution

# Force CPU backend
./evolution --backend=cpu

# Force CUDA backend (will fail if not available)
./evolution --backend=cuda

# CUDA with specific GPU
./evolution --backend=cuda --device=1

# Benchmark mode (compare CPU vs GPU)
./evolution --benchmark-backends
```

**Implementation in main.cpp**:

```cpp
#include "../../common/include/ComputeBackend.h"
#include "../../common/include/ComputeConfig.h"

int main(int argc, char** argv) {
    // Parse command-line arguments
    ComputeConfig config = ComputeConfig::loadFromFile("compute_config.yaml");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend=cpu") == 0) {
            config.backend_type = BackendType::CPU;
        } else if (strcmp(argv[i], "--backend=cuda") == 0) {
            config.backend_type = BackendType::CUDA;
        } else if (strcmp(argv[i], "--backend=auto") == 0) {
            config.backend_type = BackendType::AUTO;
        }
        // ... more options ...
    }

    // Initialize compute backend
    g_compute_backend = createComputeBackend(config.backend_type);

    if (!g_compute_backend->initialize()) {
        std::cerr << "Failed to initialize " << g_compute_backend->getName() << "\n";
        if (config.fallback_to_cpu && config.backend_type != BackendType::CPU) {
            std::cout << "Falling back to CPU backend...\n";
            g_compute_backend = createComputeBackend(BackendType::CPU);
            g_compute_backend->initialize();
        }
    }

    std::cout << "Using compute backend: " << g_compute_backend->getName() << "\n";

    // Run simulation
    // ...

    // Cleanup
    delete g_compute_backend;
}
```

---

### 8. Benchmark Mode

**Feature**: Compare CPU vs GPU side-by-side

```cpp
void runBenchmarkComparison() {
    std::cout << "Running CPU vs CUDA benchmark...\n";

    // Test with CPU
    auto cpu_backend = createComputeBackend(BackendType::CPU);
    cpu_backend->initialize();

    auto start = std::chrono::high_resolution_clock::now();
    // ... run 100 iterations ...
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start
    ).count();

    // Test with CUDA
    auto cuda_backend = createComputeBackend(BackendType::CUDA);
    if (cuda_backend->isAvailable() && cuda_backend->initialize()) {
        start = std::chrono::high_resolution_clock::now();
        // ... run 100 iterations ...
        auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();

        std::cout << "CPU:  " << cpu_time << " ms\n";
        std::cout << "CUDA: " << cuda_time << " ms\n";
        std::cout << "Speedup: " << (double)cpu_time / cuda_time << "x\n";
    } else {
        std::cout << "CUDA not available for comparison\n";
    }
}
```

---

## File Structure

```
common/
├── include/
│   ├── ComputeBackend.h        # Abstract interface
│   ├── CPUBackend.h            # CPU implementation (always)
│   ├── CUDABackend.h           # CUDA implementation (conditional)
│   ├── ComputeConfig.h         # Configuration
│   └── ComputeFactory.h        # Factory for creating backends
├── src/
│   ├── ComputeBackend.cpp      # Factory implementation
│   ├── CPUBackend.cpp          # CPU implementation
│   ├── CUDABackend.cu          # CUDA implementation (only if USE_CUDA)
│   └── ComputeConfig.cpp       # Config loading
└── config/
    └── compute_config.yaml     # User configuration

games/evolution/
├── src/
│   ├── PopulationManager.cpp   # Modified to use backend
│   └── main.cpp                # Command-line parsing
```

---

## Makefile Updates

```makefile
# User-configurable: Enable CUDA support
ENABLE_CUDA ?= 0

CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -Icommon/include

# CUDA configuration
ifeq ($(ENABLE_CUDA), 1)
    NVCC = nvcc
    CUDA_ARCH ?= sm_61  # User can override (sm_75 for RTX 2080, sm_86 for RTX 3090, etc.)
    CUDA_FLAGS = -std=c++14 -arch=$(CUDA_ARCH) -Xcompiler -fPIC
    CUDA_LIBS = -lcudart -L/usr/local/cuda/lib64
    CXXFLAGS += -DUSE_CUDA
    LDFLAGS += $(CUDA_LIBS)

    # CUDA source files
    CUDA_SOURCES = common/src/CUDABackend.cu \
                   common/src/CUDAKernels.cu

    CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

    # Rule for compiling CUDA files
    %.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
else
    CUDA_OBJECTS =
endif

# Common sources (always compiled)
COMMON_SOURCES = common/src/ComputeBackend.cpp \
                 common/src/CPUBackend.cpp \
                 common/src/ComputeConfig.cpp

COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

# Build everything
all: $(CUDA_OBJECTS) $(COMMON_OBJECTS) games

# Clean
clean:
	rm -f $(COMMON_OBJECTS) $(CUDA_OBJECTS)

# Print configuration
info:
	@echo "Build Configuration:"
	@echo "  CUDA Enabled: $(ENABLE_CUDA)"
ifeq ($(ENABLE_CUDA), 1)
	@echo "  CUDA Architecture: $(CUDA_ARCH)"
	@echo "  NVCC: $(NVCC)"
endif

.PHONY: all clean info
```

---

## Build Instructions

### CPU-Only Build (Default)

```bash
make clean
make
./games/evolution/evolution
```

### CUDA-Enabled Build

```bash
# Auto-detect GPU architecture
make clean
make ENABLE_CUDA=1

# Or specify architecture manually
make clean
make ENABLE_CUDA=1 CUDA_ARCH=sm_86  # For RTX 3090

./games/evolution/evolution
```

### Check Configuration

```bash
make info
```

Output:
```
Build Configuration:
  CUDA Enabled: 1
  CUDA Architecture: sm_86
  NVCC: nvcc
```

---

## Runtime Usage Examples

### Example 1: Auto Mode (Default)

```bash
./evolution
```

Output:
```
Loading compute configuration...
Auto-selecting compute backend...
Detected CUDA device: NVIDIA GeForce RTX 3090 (24GB)
Population size: 30 (below GPU threshold of 100)
Using compute backend: CPU (Standard)
```

### Example 2: Force CUDA

```bash
./evolution --backend=cuda
```

Output:
```
Using compute backend: CUDA (NVIDIA GeForce RTX 3090)
Device memory: 24576 MB
Compute capability: 8.6
```

### Example 3: Benchmark Comparison

```bash
./evolution --benchmark-backends
```

Output:
```
Running CPU vs CUDA benchmark (1000 iterations)...
Population: 200 rodents, 10 cats

CPU Backend:
  NN Forward: 6.2 ms/tick
  Distance Search: 8.5 ms/tick
  Mutation: 0.8 ms/tick
  Total: 15.5 ms/tick

CUDA Backend:
  NN Forward: 0.5 ms/tick (12.4x speedup)
  Distance Search: 0.3 ms/tick (28.3x speedup)
  Mutation: 0.1 ms/tick (8.0x speedup)
  Total: 0.9 ms/tick (17.2x speedup)
```

---

## Benefits of This Design

### 1. **Portability**
- Works on any system (even without CUDA)
- Graceful degradation

### 2. **Flexibility**
- Switch backends at runtime
- Compare performance easily
- Debug with CPU, deploy with GPU

### 3. **Maintainability**
- Clean separation of concerns
- Easy to add new backends (OpenCL, Metal, etc.)
- Backend-agnostic application code

### 4. **Performance**
- Zero overhead when CUDA disabled (conditionally compiled)
- Efficient GPU memory management when enabled
- Automatic selection of best backend

---

## Implementation Timeline

### Phase 1: Infrastructure (Week 1-2)
- [ ] Create abstraction layer (ComputeBackend.h)
- [ ] Implement CPU backend (always works)
- [ ] Add Makefile CUDA toggle
- [ ] Configuration system

### Phase 2: CUDA Backend (Week 3-6)
- [ ] Implement CUDABackend class
- [ ] Write CUDA kernels (distance, NN, mutation)
- [ ] Memory management
- [ ] Error handling

### Phase 3: Integration (Week 7-8)
- [ ] Modify PopulationManager to use backend
- [ ] Command-line argument parsing
- [ ] Runtime backend switching
- [ ] Benchmark mode

### Phase 4: Testing & Optimization (Week 9-10)
- [ ] Test CPU-only builds
- [ ] Test CUDA builds
- [ ] Performance tuning
- [ ] Documentation

---

## Next Steps

1. **Now**: Design and implement abstraction layer (Phase 1)
2. **After Phase 1**: Implement CPU backend (fallback)
3. **After Phase 2**: Implement CUDA backend
4. **Throughout**: Keep application code backend-agnostic

**Want me to start implementing Phase 1 (the abstraction layer)?**

