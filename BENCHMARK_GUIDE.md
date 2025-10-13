# Cursed World - CPU Baseline Benchmark Guide

This guide explains how to run and interpret CPU baseline benchmarks for the Cursed World evolution simulation.

## Overview

The benchmark system measures the performance of key computational components:

1. **Neural Network Forward Passes** - AI decision making for each agent
2. **Population Updates** - Entity updates (rodents and cats)
3. **Genetic Operations** - Evolution, sorting, mutation, offspring creation

These benchmarks establish a baseline for comparison when evaluating CUDA migration benefits.

---

## Running Benchmarks

### Method 1: Using the benchmark script (Recommended)

```bash
./run_benchmark.sh
```

Follow the on-screen instructions:
- Let the simulation run for at least 2-3 generations
- Press 'q' to quit when ready
- Results will be saved to `games/evolution/benchmark_results.txt`

### Method 2: Manual execution

```bash
cd games/evolution
./evolution
# Let it run for 2-3 generations
# Press 'q' to quit
# Check benchmark_results.txt
```

---

## Understanding the Results

### Sample Output

```
========================================
         BENCHMARK REPORT
========================================

NeuralNetwork::forward:
  Samples:  15000
  Average:  0.025 ms
  Std Dev:  0.008 ms
  Min:      0.010 ms
  Max:      0.150 ms
  Total:    375.000 s

PopulationManager::update:
  Samples:  1000
  Average:  15.234 ms
  Std Dev:  2.345 ms
  Min:      12.000 ms
  Max:      25.000 ms
  Total:    15.234 s

PopulationManager::update_rodents:
  Samples:  1000
  Average:  10.123 ms
  Std Dev:  1.567 ms
  Min:      8.000 ms
  Max:      18.000 ms
  Total:    10.123 s

PopulationManager::update_cats:
  Samples:  1000
  Average:  0.456 ms
  Std Dev:  0.123 ms
  Min:      0.300 ms
  Max:      2.000 ms
  Total:    0.456 s

PopulationManager::evolveGeneration:
  Samples:  3
  Average:  145.678 ms
  Std Dev:  12.345 ms
  Min:      130.000 ms
  Max:      160.000 ms
  Total:    0.437 s

PopulationManager::evolve_sort:
  Samples:  3
  Average:  2.345 ms
  Std Dev:  0.234 ms
  Min:      2.000 ms
  Max:      2.700 ms
  Total:    0.007 s

PopulationManager::evolve_create_offspring:
  Samples:  3
  Average:  120.456 ms
  Std Dev:  10.123 ms
  Min:      110.000 ms
  Max:      135.000 ms
  Total:    0.361 s

NeuralNetwork::mutate:
  Samples:  240
  Average:  0.456 ms
  Std Dev:  0.089 ms
  Min:      0.300 ms
  Max:      1.200 ms
  Total:    0.109 s

========================================
```

---

## Key Metrics Explained

### 1. Neural Network Forward Pass
- **What it measures**: Time to compute one neural network inference
- **Sample count**: Number of agents × number of ticks
- **Typical value**: 0.01-0.05 ms per forward pass
- **CUDA opportunity**: ⭐⭐⭐⭐⭐ Excellent candidate for GPU acceleration

**Analysis**:
- With ~30 rodents + 3 cats = 33 agents per tick
- At ~1000 ticks/generation × 3 generations = 3000 ticks
- Expected samples: ~99,000 forward passes
- If average is 0.025 ms → Total: ~2.5 seconds

### 2. Population Update
- **What it measures**: Time to update all entities per tick
- **Includes**: Rodent updates + Cat updates + collision detection
- **Sample count**: Number of simulation ticks
- **Typical value**: 10-20 ms per tick
- **CUDA opportunity**: ⭐⭐⭐⭐ Good candidate for parallelization

**Analysis**:
- Dominated by neural network forward passes
- Also includes movement, collision detection, eating
- With 33 agents doing ~0.025ms each → ~0.8ms just for inference
- Remaining time is game logic (10-15ms)

### 3. Genetic Evolution
- **What it measures**: Time to evolve to next generation
- **Includes**: Sorting, parent selection, offspring creation, mutation
- **Sample count**: Number of generations
- **Typical value**: 100-200 ms per generation
- **CUDA opportunity**: ⭐⭐⭐ Moderate benefit

**Subcomponents**:
- **Sorting**: 2-5 ms (small population, quick sort)
- **Offspring creation**: 100-150 ms (includes mutation)
- **Mutation**: 0.3-0.5 ms per agent (~80 offspring → 40ms total)

---

## Performance Targets for CUDA Migration

Based on typical CPU baseline results:

| Component | CPU Baseline | CUDA Target | Expected Speedup |
|-----------|--------------|-------------|------------------|
| **Neural Network Forward** | 0.025 ms | 0.002 ms | 10-15x |
| **Population Update** | 15 ms | 3-5 ms | 3-5x |
| **Genetic Evolution** | 150 ms | 30 ms | 5x |
| **Overall Tick Rate** | 15-20 ms/tick | 4-6 ms/tick | 3-4x |

### Estimated Performance Gains

**Current Performance (30 rodents, 3 cats, 2000 ticks/gen)**:
- Tick rate: ~15-20 ms/tick
- Generation time: ~30-40 seconds
- 100 generations: ~50-65 minutes

**Expected CUDA Performance (same parameters)**:
- Tick rate: ~4-6 ms/tick
- Generation time: ~8-12 seconds
- 100 generations: ~13-20 minutes

**Speedup**: ~3-4x overall simulation speed

### Scaled Up Performance (1000 rodents, 30 cats)

**Current CPU (estimated)**:
- Tick rate: ~200-250 ms/tick
- Generation time: ~6-8 minutes
- Likely impractical for long runs

**Expected CUDA**:
- Tick rate: ~10-20 ms/tick
- Generation time: ~20-40 seconds
- **Speedup**: 10-20x (better GPU utilization with more agents)

---

## Bottleneck Analysis

### What's taking the most time?

Run this analysis on your benchmark results:

1. **Neural Network Forward Passes**
   ```
   Total time = Average × Sample count
   Example: 0.025 ms × 99,000 samples = 2,475 ms (2.5 seconds)
   ```

2. **Per-Tick Overhead**
   ```
   Update time - (NN time × agents)
   Example: 15 ms - (0.025 ms × 33) = 14.2 ms
   ```
   This overhead is game logic (movement, collision, rendering)

3. **Evolution Overhead**
   ```
   Evolution time - (Mutation time × offspring count)
   Example: 150 ms - (0.45 ms × 80) = 114 ms
   ```
   This includes sorting (2-3ms) and position finding (~110ms)

### Where CUDA Helps Most

1. **Neural networks** (⭐⭐⭐⭐⭐): Batch inference is highly parallel
2. **Mutations** (⭐⭐⭐⭐): Independent weight updates
3. **Fitness calculations** (⭐⭐⭐): Parallel evaluation
4. **Sorting** (⭐⭐⭐): GPU sorting libraries (Thrust)

### Where CUDA Helps Less

1. **Position finding**: Dynamic, branching logic
2. **Collision detection**: Small-scale spatial queries
3. **Memory allocation**: Dynamic population changes
4. **Terminal rendering**: CPU-bound, ncurses-based

---

## Next Steps

After collecting baseline results:

1. **Document your actual numbers** in this file
2. **Identify bottlenecks** using the analysis above
3. **Compare with targets** to validate CUDA migration value
4. **Proceed with CUDA implementation** if speedup > 3x is expected

---

## Sample Collection Commands

To collect comprehensive data, run multiple benchmark sessions:

```bash
# Short run (1 generation) - quick validation
./run_benchmark.sh
# Let run for 1 generation, quit

# Medium run (3-5 generations) - typical workload
./run_benchmark.sh
# Let run for 3-5 generations, quit

# Long run (10 generations) - stress test
./run_benchmark.sh
# Let run for 10 generations, quit
```

Compare results across runs to check for:
- Performance degradation over time
- Memory leaks (check resident memory)
- Consistency (std deviation should be low)

---

## Your Benchmark Results

**Date**: ___________
**Hardware**: ___________
**Population**: ___ rodents, ___ cats
**Generations tested**: ___

### Key Findings:

| Metric | Value | Notes |
|--------|-------|-------|
| NN Forward (avg) | ___ ms | |
| Update (avg) | ___ ms | |
| Evolution (avg) | ___ ms | |
| Tick rate | ___ ms/tick | |
| Generation time | ___ seconds | |

### Conclusion:

Is CUDA migration worth it?
☐ Yes - Expected speedup: ___x
☐ No - Current performance is acceptable
☐ Maybe - Need more data

---

**Generated**: 2025-10-11
