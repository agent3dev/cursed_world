#!/bin/bash
# Benchmark runner for Cursed World Evolution Simulation
# This script runs the evolution simulation and collects performance metrics

echo "=========================================="
echo "   CURSED WORLD BENCHMARK RUNNER"
echo "=========================================="
echo ""
echo "This will run the evolution simulation to collect performance metrics."
echo "The simulation will run interactively. To get meaningful benchmark data:"
echo ""
echo "  1. Let it run for at least 2-3 generations (watch the 'Gen:' counter)"
echo "  2. Press 'q' to quit when you want to stop"
echo "  3. Benchmark results will be displayed and saved to benchmark_results.txt"
echo ""
echo "Benchmarks collected:"
echo "  - Neural network forward passes (per agent per tick)"
echo "  - Neural network mutations (during evolution)"
echo "  - Population updates (cats and rodents)"
echo "  - Genetic operations (sorting, offspring creation)"
echo ""
echo "Press Enter to start the simulation..."
read

cd games/evolution
./evolution

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "Results saved to: games/evolution/benchmark_results.txt"
echo "=========================================="
