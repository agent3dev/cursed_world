# Phase 2: Integration Guide

**Status**: Phase 1 Complete ✅ - Ready to begin Phase 2
**Next Step**: Integrate compute backend with PopulationManager

---

## Quick Status Check

### ✅ Phase 1 Complete
- [x] Abstract ComputeBackend interface
- [x] CPUBackend implementation (fully functional)
- [x] CUDABackend infrastructure (device detection ready)
- [x] ComputeConfig system (YAML + command-line)
- [x] Factory pattern for backend creation
- [x] Makefiles updated with ENABLE_CUDA flag
- [x] **Build successful** (3.3MB binary)

### Files Created (Phase 1)
```
common/include/
  ├── ComputeBackend.h    (4.3K) - Abstract interface
  ├── CPUBackend.h        (1.9K) - CPU implementation
  ├── CUDABackend.h       (4.0K) - CUDA infrastructure
  └── ComputeConfig.h     (3.6K) - Configuration system

common/src/
  ├── ComputeBackend.cpp  (3.3K) - Factory function
  ├── CPUBackend.cpp      (3.8K) - CPU implementation
  ├── CUDABackend.cpp     (6.3K) - CUDA stubs
  └── ComputeConfig.cpp   (6.1K) - Config parser

Root:
  ├── compute_config.yaml (0.5K) - Example configuration
  └── CUDA_TOGGLEABLE_PHASE1_COMPLETE.md - Full documentation
```

---

## Phase 2 Overview: Integration (Week 3-6)

### Goal
Integrate the compute backend with PopulationManager to enable batched neural network operations and distance searches.

### Key Tasks

#### 1. Add Backend to PopulationManager
**File**: `games/evolution/include/PopulationManager.h`

Add private member:
```cpp
#include "../../common/include/ComputeBackend.h"

class PopulationManager {
private:
    std::unique_ptr<ComputeBackend> compute_backend_;
    // ... existing members
```

#### 2. Initialize Backend in Constructor
**File**: `games/evolution/src/PopulationManager.cpp`

```cpp
PopulationManager::PopulationManager(...)
    : ... {

    // Create compute backend (defaults to AUTO)
    compute_backend_ = createComputeBackend(BackendType::AUTO, true);

    if (!compute_backend_) {
        std::cerr << "[PopulationManager] Failed to create compute backend!\n";
        // Fall back to direct method calls (existing code)
    } else {
        std::cout << "[PopulationManager] Using backend: "
                  << compute_backend_->getName() << "\n";
    }
}
```

#### 3. Batch Neural Network Forward Passes
**Current code** (in `PopulationManager::update()`):
```cpp
// Each rodent calls brain.forward() individually
for (auto& rodent : population) {
    if (rodent->isAlive()) {
        rodent->update(matrix);  // Calls brain.forward() internally
    }
}
```

**New batched approach**:
```cpp
// Collect all inputs
std::vector<std::vector<double>> inputs;
std::vector<NeuralNetwork*> networks;
std::vector<Rodent*> active_rodents;

for (auto& rodent : population) {
    if (rodent->isAlive()) {
        inputs.push_back(rodent->getSurroundingInfo(matrix));
        networks.push_back(&rodent->getBrain());
        active_rodents.push_back(rodent.get());
    }
}

// Batch process on GPU/CPU
std::vector<std::vector<double>> outputs;
compute_backend_->batchedForward(inputs, networks, outputs);

// Apply outputs to rodents
for (size_t i = 0; i < active_rodents.size(); i++) {
    active_rodents[i]->applyNeuralNetworkOutput(outputs[i], matrix);
}
```

#### 4. Batch Distance Searches
**Current code** (in `Rodent::update()`):
```cpp
// Each rodent searches individually
NearestEntity nearestCat = findNearestCat(matrix, 20);
NearestEntity nearestPeer = findNearestPeer(matrix, 15);
```

**New batched approach**:
```cpp
// In PopulationManager::update()

// Build agent and target position lists
std::vector<AgentPosition> rodent_positions;
std::vector<AgentPosition> cat_positions;

for (auto& rodent : population) {
    if (rodent->isAlive()) {
        rodent_positions.push_back({rodent->getX(), rodent->getY(), rodent->getId()});
    }
}

for (auto& cat : cats) {
    if (cat->isAlive()) {
        cat_positions.push_back({cat->getX(), cat->getY(), cat->getId()});
    }
}

// Batch search for nearest cats (all rodents at once)
std::vector<NearestEntityResult> nearest_cats;
compute_backend_->findNearestEntities(rodent_positions, cat_positions, nearest_cats, 20);

// Store results for later use
for (size_t i = 0; i < rodent_positions.size(); i++) {
    rodents[i]->setNearestCat(nearest_cats[i]);
}
```

#### 5. Add Setter Methods to Rodent/Cat
**File**: `games/evolution/include/Rodent.h`

```cpp
class Rodent : public Actuator {
private:
    NearestEntity cached_nearest_cat_;
    NearestEntity cached_nearest_peer_;
    NearestEntity cached_nearest_food_;

public:
    void setCachedNearestCat(const NearestEntity& entity) {
        cached_nearest_cat_ = entity;
    }

    void setCachedNearestPeer(const NearestEntity& entity) {
        cached_nearest_peer_ = entity;
    }

    void setCachedNearestFood(const NearestEntity& entity) {
        cached_nearest_food_ = entity;
    }

    // Modify getSurroundingInfo() to use cached values
    std::vector<double> getSurroundingInfo(TerminalMatrix& matrix);
};
```

#### 6. Integrate with Main Simulation Loop
**File**: `games/evolution/src/main.cpp`

```cpp
#include "../../common/include/ComputeConfig.h"

int main(int argc, char* argv[]) {
    // Parse configuration
    ComputeConfig config;
    config.parseCommandLine(argc, argv);

    // Pass config to PopulationManager
    PopulationManager pop_manager(config);

    // ... rest of simulation
}
```

---

## Implementation Strategy

### Step 1: Add Backend Member (Week 3)
- Add `compute_backend_` to PopulationManager
- Initialize in constructor
- Keep existing code paths as fallback
- **Test**: Build and run, verify backend initialization

### Step 2: Batch Neural Networks (Week 4)
- Collect inputs from all agents
- Call `batchedForward()`
- Apply outputs back to agents
- **Test**: Verify same behavior as before

### Step 3: Batch Distance Searches (Week 5)
- Collect agent/target positions
- Call `findNearestEntities()`
- Cache results in agents
- **Test**: Verify same behavior, measure performance

### Step 4: Command-Line Integration (Week 6)
- Add command-line parsing to main.cpp
- Add configuration file loading
- Add performance statistics output
- **Test**: Try all backend options, verify fallback

---

## Testing Plan

### Test 1: Build Verification
```bash
cd /home/erza/develop/cursed_world/games/evolution
make clean && make -j4
./evolution
# Should run normally with CPU backend
```

### Test 2: Backend Selection
```bash
./evolution --backend=cpu
# Verify uses CPU backend

./evolution --backend=auto
# Verify auto-selects based on population
```

### Test 3: Configuration File
```bash
./evolution --config=../../compute_config.yaml
# Verify loads settings from file
```

### Test 4: Performance Comparison
```bash
# Run 10 generations with CPU
./evolution --backend=cpu
# Note generation time

# Run 10 generations with AUTO
./evolution --backend=auto
# Should be similar (CUDA kernels not implemented yet)
```

---

## Expected Outcomes (Phase 2)

### After Integration
- ✅ Simulation uses compute backend for all operations
- ✅ Can switch backends via command-line
- ✅ Can load configuration from YAML
- ✅ Performance roughly the same (both use CPU)
- ✅ Ready for CUDA kernel implementation (Phase 3)

### Performance (Phase 2 Complete)
**Both backends use CPU**, so performance should be identical:
- CPU backend: ~40ms/tick (200 agents)
- CUDA backend: ~40ms/tick (200 agents) - using CPU fallback
- **Speedup: 1.0×** (expected)

### Performance (Phase 3 Complete - with GPU kernels)
Once CUDA kernels are implemented:
- CPU backend: ~40ms/tick (200 agents)
- CUDA backend: ~8ms/tick (200 agents)
- **Speedup: 5.0×** ✅

---

## Phase 3 Preview: CUDA Kernels (Week 7-8)

After Phase 2 integration is complete, Phase 3 will implement actual GPU kernels:

### Kernel 1: Neural Network Forward Pass
```cuda
__global__ void batchedForwardKernel(
    double* d_inputs,      // [batch_size][input_dim]
    double* d_weights,     // [weight_count]
    double* d_outputs,     // [batch_size][output_dim]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Process neural network for agent idx
        // ... matrix multiplication ...
    }
}
```

### Kernel 2: Distance Search
```cuda
__global__ void findNearestEntitiesKernel(
    AgentPosition* d_agents,      // [agent_count]
    AgentPosition* d_targets,     // [target_count]
    NearestEntityResult* d_results, // [agent_count]
    int agent_count,
    int target_count,
    int search_radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < agent_count) {
        // Find nearest target for agent idx
        // Parallelized O(N) search per agent
    }
}
```

### Kernel 3: Mutation
```cuda
__global__ void batchedMutateKernel(
    double* d_weights,        // [total_weights]
    double* d_random,         // [total_weights] - pre-generated random
    double mutation_rate,
    double mutation_amount,
    int weight_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_count) {
        if (d_random[idx] < mutation_rate) {
            d_weights[idx] += d_random[idx + weight_count] * mutation_amount;
        }
    }
}
```

---

## Documentation

### Read These First
1. `CUDA_TOGGLEABLE_PHASE1_COMPLETE.md` - Phase 1 summary
2. `CUDA_TOGGLEABLE_DESIGN.md` - Full architecture design
3. `compute_config.yaml` - Configuration example

### Reference
- `common/include/ComputeBackend.h` - Interface documentation
- `common/include/CPUBackend.h` - CPU backend
- `common/include/CUDABackend.h` - CUDA backend

---

## Quick Commands

```bash
# Clean build
make clean && make -j4

# Build with CUDA support (for testing Phase 3)
make clean && make ENABLE_CUDA=1 -j4

# Run with CPU backend
./evolution --backend=cpu

# Run with AUTO selection
./evolution --backend=auto --auto-threshold=100

# Load configuration
./evolution --config=../../compute_config.yaml

# Show configuration
./evolution --backend=auto --help  # (to be implemented in Phase 2)
```

---

## Timeline

- ✅ **Week 1-2**: Phase 1 - Infrastructure - **COMPLETE**
- ⏳ **Week 3**: Add backend to PopulationManager
- ⏳ **Week 4**: Batch neural network operations
- ⏳ **Week 5**: Batch distance searches
- ⏳ **Week 6**: Command-line integration + testing
- ⏳ **Week 7-8**: Phase 3 - CUDA kernels
- ⏳ **Week 9-10**: Phase 4 - Optimization

**Total**: 10 weeks to full GPU acceleration

---

## Summary

✅ **Phase 1: Complete and ready**
- Infrastructure in place
- Building successfully
- Both backends compile
- Configuration system working

⏳ **Phase 2: Up next**
- Integrate with PopulationManager
- Batch operations for parallelization
- Command-line argument parsing
- Performance remains same (CPU)

🚀 **Phase 3: Future**
- Implement CUDA kernels
- Achieve 5-20× speedup
- GPU memory optimization

---

**Ready to begin Phase 2 integration!**

For questions or issues, refer to:
- `CUDA_TOGGLEABLE_PHASE1_COMPLETE.md`
- `CUDA_TOGGLEABLE_DESIGN.md`
