# Smell Feature Design: Distance-Based Sensing
## Neural Network Enhancement for Cursed World Evolution

**Date**: 2025-10-11
**Feature**: Add long-range sensing capabilities to rodents and cats

---

## Overview

Currently, agents can only see their immediate 8 neighbors (1 tile away). The "smell" feature adds long-range distance sensing, allowing agents to detect:

1. **Nearest predator** (for rodents) - distance & direction
2. **Nearest prey** (for cats) - distance & direction
3. **Nearest peer** (same species) - distance & direction
4. **Nearest food** (for rodents) - distance & direction

This significantly increases neural network complexity and makes CUDA migration much more valuable.

---

## Current State

### Rodent Neural Network (9 inputs)
```
Input 0-7: 8 surrounding tiles (immediate neighbors)
Input 8:   Energy level (0-1)
```

### Cat Neural Network (10 inputs)
```
Input 0-7: 8 surrounding tiles (immediate neighbors)
Input 8:   Rodents eaten count (0-1)
Input 9:   Eat cooldown (0-1)
```

**Problem**: Agents are "blind" beyond 1 tile. They stumble around randomly and only react when threats/food are adjacent.

---

## Proposed Enhancement

### New Rodent Network (17 inputs = 9 + 8)

```
// Existing (9 inputs)
Input 0-7:   8 surrounding tiles
Input 8:     Energy level

// NEW: Smell sensors (8 inputs)
Input 9-10:  Nearest CAT (distance_x, distance_y) normalized
Input 11:    Nearest CAT Manhattan distance (normalized)
Input 12-13: Nearest RODENT peer (distance_x, distance_y)
Input 14:    Nearest RODENT peer distance
Input 15-16: Nearest FOOD (distance_x, distance_y)
```

**Total**: 17 inputs (9 → 17 = 1.9× increase)

### New Cat Network (16 inputs = 10 + 6)

```
// Existing (10 inputs)
Input 0-7:   8 surrounding tiles
Input 8:     Rodents eaten
Input 9:     Eat cooldown

// NEW: Smell sensors (6 inputs)
Input 10-11: Nearest RODENT (distance_x, distance_y)
Input 12:    Nearest RODENT Manhattan distance
Input 13-14: Nearest CAT peer (distance_x, distance_y)
Input 15:    Nearest CAT peer distance
```

**Total**: 16 inputs (10 → 16 = 1.6× increase)

---

## Performance Impact

### Computation Increase

**Rodent Network** (9→32→9 → 17→32→9):
```
Old FLOPs:
  Input→Hidden:  9 × 32  = 288
  Recurrent:    32 × 32  = 1,024
  Hidden→Output: 32 × 9  = 288
  Total: 1,600 FLOPs

New FLOPs:
  Input→Hidden:  17 × 32 = 544
  Recurrent:    32 × 32  = 1,024
  Hidden→Output: 32 × 9  = 288
  Total: 1,856 FLOPs (1.16× increase)
```

**But**: Distance calculation overhead (scanning for nearest entities)

**Distance search algorithm**:
```cpp
// For each agent:
//   Scan all peers (30 rodents / 3 cats)
//   Calculate Manhattan distance
//   Find minimum
//
// Worst case: O(N²) where N = population
// Per rodent: ~33 distance calculations
// Per tick: 30 rodents × 33 checks = 990 calculations
```

**Total per-rodent time**:
```
Old: 0.025 ms (NN forward only)
New: 0.025 ms (NN) + 0.015 ms (distance search) = 0.040 ms
```

---

## CUDA Impact Analysis

### Small Population (30 rodents, 3 cats)

**Without smell**:
- NN time: 0.8 ms/tick
- Game logic: 14 ms/tick
- Total: 14.8 ms/tick
- With CUDA: 14.08 ms/tick (**1.05× speedup**) ❌

**With smell**:
- NN time: 1.0 ms/tick (1.16× increase)
- Distance search: 0.5 ms/tick (O(N²))
- Game logic: 14 ms/tick
- Total: 15.5 ms/tick
- With CUDA: 14.1 ms/tick (**1.10× speedup**) ❌

**Verdict**: Still not worth it for small populations.

---

### Medium Population (200 rodents, 10 cats)

**Without smell**:
- NN time: 5.3 ms/tick
- Game logic: 8 ms/tick
- Total: 13.3 ms/tick
- With CUDA: 8.5 ms/tick (**1.56× speedup**) ⚠️

**With smell**:
- NN time: 6.2 ms/tick (1.16× increase)
- Distance search: 8.5 ms/tick (O(N²), 210² = 44,100 checks)
- Game logic: 8 ms/tick
- Total: 22.7 ms/tick
- With CUDA (both NN + distance): 8.5 ms/tick (**2.67× speedup**) ✅

**Verdict**: Now CUDA is worthwhile! The O(N²) distance search is parallelizable.

---

### Large Population (1000 rodents, 30 cats)

**Without smell**:
- NN time: 25.8 ms/tick
- Game logic: 5 ms/tick
- Total: 30.8 ms/tick
- With CUDA: 7 ms/tick (**4.4× speedup**) ✅

**With smell**:
- NN time: 30 ms/tick (1.16× increase)
- Distance search: 130 ms/tick (O(N²), 1030² = 1.06M checks!!)
- Game logic: 5 ms/tick
- Total: 165 ms/tick
- With CUDA (both NN + distance): 8 ms/tick (**20.6× speedup**) ✅✅✅

**Verdict**: CUDA is essential! CPU performance collapses due to O(N²) searches.

---

## Key Insight: O(N²) Distance Search

**The smell feature makes CUDA migration a no-brainer for medium/large populations!**

### Why Distance Search is Expensive

For each agent, finding nearest peer/predator/prey requires:
1. Loop through all entities of target type
2. Calculate Manhattan distance
3. Track minimum

**Cost**: O(N) per agent → O(N²) total

For 1000 agents:
- 1000 rodents checking for nearest cat among 30 cats = 30,000 checks
- 1000 rodents checking for nearest peer = 1,000,000 checks
- 1000 rodents checking for nearest food among ~200 food = 200,000 checks
- **Total: ~1.23 million distance calculations per tick**

At ~0.1 microseconds per distance calc: **~130 ms/tick just for distance search!**

---

## CUDA Optimization Strategies

### Strategy 1: Parallel Distance Calculations (Easy Win)

```cuda
__global__ void findNearestEntities(
    int* agent_positions_x,      // [N_agents]
    int* agent_positions_y,      // [N_agents]
    int* target_positions_x,     // [N_targets]
    int* target_positions_y,     // [N_targets]
    int* nearest_dx,             // [N_agents] output
    int* nearest_dy,             // [N_agents] output
    int* nearest_dist,           // [N_agents] output
    int N_agents,
    int N_targets
) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= N_agents) return;

    int my_x = agent_positions_x[agent_id];
    int my_y = agent_positions_y[agent_id];

    int min_dist = 999999;
    int best_dx = 0;
    int best_dy = 0;

    // Each thread finds its own nearest target
    for (int t = 0; t < N_targets; t++) {
        int dx = target_positions_x[t] - my_x;
        int dy = target_positions_y[t] - my_y;
        int dist = abs(dx) + abs(dy);  // Manhattan distance

        if (dist < min_dist) {
            min_dist = dist;
            best_dx = dx;
            best_dy = dy;
        }
    }

    nearest_dx[agent_id] = best_dx;
    nearest_dy[agent_id] = best_dy;
    nearest_dist[agent_id] = min_dist;
}
```

**Speedup**: ~100-200× for large populations (embarrassingly parallel)

**CPU**: 130 ms for 1000 agents
**GPU**: ~1 ms for 1000 agents

---

### Strategy 2: Spatial Hashing (Advanced)

For very large populations (10,000+), even GPU brute-force becomes slow. Use spatial hashing:

```cuda
// Divide world into grid cells
// Hash each entity to cell
// Only search nearby cells for nearest neighbors
// Reduces O(N²) to O(N × k) where k = avg entities per cell
```

**Speedup**: Additional 5-10× for populations > 5000

---

## Implementation Plan

### Phase 1: CPU Implementation (1 week)

**1.1: Add distance search functions** (2 days)

```cpp
// In Rodent.cpp
struct NearestEntity {
    int dx, dy;  // Direction vector
    int distance;  // Manhattan distance
    bool found;
};

NearestEntity findNearestCat(TerminalMatrix& matrix, int search_radius = 20);
NearestEntity findNearestPeer(TerminalMatrix& matrix, int search_radius = 15);
NearestEntity findNearestFood(TerminalMatrix& matrix, int search_radius = 15);
```

**1.2: Update neural network inputs** (2 days)

```cpp
// Modify Rodent::getSurroundingInfo()
std::vector<double> Rodent::getSurroundingInfo(TerminalMatrix& matrix) {
    std::vector<double> info;

    // Existing: 8 surrounding tiles
    // ... existing code ...

    // NEW: Smell inputs
    auto nearestCat = findNearestCat(matrix);
    info.push_back(nearestCat.dx / 20.0);  // Normalize to -1 to +1
    info.push_back(nearestCat.dy / 20.0);
    info.push_back(nearestCat.distance / 20.0);  // 0 to 1

    auto nearestPeer = findNearestPeer(matrix);
    info.push_back(nearestPeer.dx / 15.0);
    info.push_back(nearestPeer.dy / 15.0);
    info.push_back(nearestPeer.distance / 15.0);

    auto nearestFood = findNearestFood(matrix);
    info.push_back(nearestFood.dx / 15.0);
    info.push_back(nearestFood.dy / 15.0);

    return info;  // Now 17 inputs instead of 9
}
```

**1.3: Update NN architecture** (1 day)

```cpp
// In Rodent.cpp constructor
Rodent::Rodent(...)
    : brain({17, 32, 9})  // Changed from {9, 16, 9}
{
    // ...
}

// In Cat.cpp constructor
Cat::Cat(...)
    : brain({16, 32, 9})  // Changed from {10, 16, 9}
{
    // ...
}
```

**1.4: Update brain files** (1 day)

- Delete old `best_brain.dat` and `best_cat_brain.dat`
- Let evolution train new brains with larger input size

---

### Phase 2: Benchmark New Performance (1 day)

```bash
./run_benchmark.sh
# Let run for 3-5 generations
# Document new timings
```

**Expected results**:
- Small population: ~1.1× slower (more inputs)
- Medium population: ~1.7× slower (O(N²) distance search)
- Large population: ~5× slower (O(N²) kills performance)

This makes the case for CUDA even stronger!

---

### Phase 3: CUDA Optimization (4-6 weeks)

**Week 1-2**: Batched NN inference (existing plan)
**Week 3**: Parallel distance search kernel
**Week 4**: Integration and testing
**Week 5-6**: Optimization and profiling

---

## Expected Final Performance

### Large Population (1000 agents) with Smell

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| NN Forward Pass | 30 ms | 2 ms | 15× |
| Distance Search | 130 ms | 1 ms | 130× |
| Game Logic | 5 ms | 5 ms | 1× |
| **Total** | **165 ms** | **8 ms** | **20.6×** |

**Per generation** (2000 ticks):
- CPU: 330 seconds = 5.5 minutes
- GPU: 16 seconds
- **20× faster**

**100 generations**:
- CPU: 9.2 hours
- GPU: 27 minutes
- **Time saved: 8.7 hours per experiment**

---

## Biological Realism Benefits

### Why This Makes Sense

**Real animals use multiple senses**:
1. **Vision**: Immediate neighbors (current 8-tile input)
2. **Smell**: Long-range detection (new distance inputs)
3. **Internal state**: Energy, hunger (current energy input)

**Emergent behaviors with smell**:
- Rodents flee from approaching cats (not just adjacent)
- Cats hunt strategically (chase fleeing rodents)
- Rodents cooperate by staying near peers (safety in numbers)
- Cats avoid each other (territorial behavior)

---

## Code Changes Summary

### Files to Modify

| File | Changes | LOC |
|------|---------|-----|
| `Rodent.h` | Add helper methods | +10 |
| `Rodent.cpp` | Implement distance search, update getSurroundingInfo() | +80 |
| `Cat.h` | Add helper methods | +10 |
| `Cat.cpp` | Implement distance search, update getSurroundingInfo() | +70 |
| `NeuralNetwork` config | Update input layer sizes | ~5 |

**Total**: ~175 LOC for CPU implementation

### CUDA Files to Add (later)

| File | Purpose | LOC |
|------|---------|-----|
| `common/include/DistanceKernels.h` | Distance search kernels | ~100 |
| `common/src/DistanceKernels.cu` | Implementation | ~200 |
| `common/include/GPUEntityManager.h` | Entity position management | ~150 |
| `common/src/GPUEntityManager.cu` | Implementation | ~300 |

**Total**: ~750 LOC for CUDA implementation

---

## Decision Matrix

### Should You Add Smell Feature?

**✅ YES if**:
- You want smarter, more realistic agent behavior
- You plan to scale to 200+ agents
- You're willing to accept slower CPU performance initially
- You're considering CUDA migration (smell makes CUDA essential)

**⚠️ MAYBE if**:
- Current agent intelligence is acceptable
- Population will stay at 30-100 agents
- Not planning CUDA migration soon

**❌ NO if**:
- Performance is already borderline
- Need real-time interaction
- Population < 50 agents permanently

---

## Recommended Approach

### Option A: Incremental (Conservative)

1. **Week 1**: Add smell to Rodents only
2. **Week 2**: Test and tune (observe behavior changes)
3. **Week 3**: Add smell to Cats
4. **Week 4**: Benchmark and decide on CUDA

**Pros**: Low risk, can evaluate benefits at each step
**Cons**: Slower progress

### Option B: Full Implementation (Aggressive)

1. **Day 1-3**: Implement smell for both species
2. **Day 4-5**: Test and fix bugs
3. **Day 6-7**: Benchmark and analyze
4. **Week 2+**: Start CUDA migration if justified

**Pros**: Fast iteration, immediate data
**Cons**: More debugging

### Option C: Smell + CUDA Together (Optimal)

1. **Week 1**: Implement smell feature on CPU
2. **Week 2**: Benchmark and confirm O(N²) bottleneck
3. **Week 3-8**: Implement CUDA with both NN + distance search
4. **Week 9-10**: Testing and optimization

**Pros**: Maximum performance gain, solves O(N²) problem immediately
**Cons**: Longest initial timeline, highest complexity

---

## Recommendation

**I recommend Option C: Smell + CUDA Together**

**Rationale**:
1. Smell feature adds O(N²) complexity that makes CPU impractical for medium/large populations
2. CUDA distance search is trivial to implement (simple parallel loop)
3. Combined speedup is 20× vs 5× for NN alone
4. The O(N²) bottleneck is a perfect showcase for GPU acceleration

**Timeline**: 10 weeks for complete implementation
**ROI**: 20× speedup on realistic workloads (200+ agents)

---

## Next Steps

1. **Decide**: Which option (A, B, or C)?

2. **If Option A or B** (CPU-only smell):
   ```bash
   # I can implement the CPU version now (Phase 1)
   # Estimated time: 1 week
   # Expected result: Smarter agents, 2-5× slower
   ```

3. **If Option C** (Smell + CUDA):
   ```bash
   # Phase 1: CPU smell (1 week)
   # Phase 2: Benchmark (1 day)
   # Phase 3: CUDA implementation (8 weeks)
   # Expected result: Smarter agents, 10-20× faster
   ```

**Want me to start implementing the smell feature now?**

