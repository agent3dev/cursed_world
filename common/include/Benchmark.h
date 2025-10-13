#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// High-resolution timer for benchmarking
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    Timer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    // Get elapsed time in microseconds
    long long getMicroseconds() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time).count();
    }

    // Get elapsed time in milliseconds
    double getMilliseconds() const {
        return getMicroseconds() / 1000.0;
    }

    // Get elapsed time in seconds
    double getSeconds() const {
        return getMicroseconds() / 1000000.0;
    }
};

// Benchmark statistics collector
class BenchmarkStats {
private:
    struct Stats {
        std::vector<double> samples;  // In microseconds
        long long total_us;
        size_t count;
        double min_us;
        double max_us;

        Stats() : total_us(0), count(0), min_us(1e9), max_us(0) {}

        void add(double microseconds) {
            samples.push_back(microseconds);
            total_us += microseconds;
            count++;
            if (microseconds < min_us) min_us = microseconds;
            if (microseconds > max_us) max_us = microseconds;
        }

        double getAverage() const {
            return count > 0 ? total_us / (double)count : 0.0;
        }

        double getStdDev() const {
            if (count == 0) return 0.0;
            double avg = getAverage();
            double sum_sq_diff = 0.0;
            for (double sample : samples) {
                double diff = sample - avg;
                sum_sq_diff += diff * diff;
            }
            return std::sqrt(sum_sq_diff / count);
        }
    };

    std::map<std::string, Stats> stats_map;

public:
    void recordSample(const std::string& name, double microseconds) {
        stats_map[name].add(microseconds);
    }

    void clear() {
        stats_map.clear();
    }

    void printReport(std::ostream& out = std::cout) const {
        out << "\n========================================\n";
        out << "         BENCHMARK REPORT\n";
        out << "========================================\n\n";

        for (const auto& entry : stats_map) {
            const std::string& name = entry.first;
            const Stats& stats = entry.second;

            out << name << ":\n";
            out << "  Samples:  " << stats.count << "\n";
            out << "  Average:  " << std::fixed << std::setprecision(3) << stats.getAverage() / 1000.0 << " ms\n";
            out << "  Std Dev:  " << std::fixed << std::setprecision(3) << stats.getStdDev() / 1000.0 << " ms\n";
            out << "  Min:      " << std::fixed << std::setprecision(3) << stats.min_us / 1000.0 << " ms\n";
            out << "  Max:      " << std::fixed << std::setprecision(3) << stats.max_us / 1000.0 << " ms\n";
            out << "  Total:    " << std::fixed << std::setprecision(3) << stats.total_us / 1000000.0 << " s\n";
            out << "\n";
        }

        out << "========================================\n";
    }

    void saveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            printReport(file);
            file.close();
        }
    }

    // Get statistics for a specific benchmark
    bool getStats(const std::string& name, double& avg_ms, double& min_ms, double& max_ms, size_t& count) const {
        auto it = stats_map.find(name);
        if (it == stats_map.end()) return false;

        const Stats& stats = it->second;
        avg_ms = stats.getAverage() / 1000.0;
        min_ms = stats.min_us / 1000.0;
        max_ms = stats.max_us / 1000.0;
        count = stats.count;
        return true;
    }
};

// RAII-style scoped timer
class ScopedTimer {
private:
    Timer timer;
    std::string name;
    BenchmarkStats* stats;

public:
    ScopedTimer(const std::string& benchmark_name, BenchmarkStats* benchmark_stats = nullptr)
        : name(benchmark_name), stats(benchmark_stats) {
        timer.start();
    }

    ~ScopedTimer() {
        timer.stop();
        if (stats) {
            stats->recordSample(name, timer.getMicroseconds());
        }
    }

    // Get elapsed time without stopping the timer
    double getElapsedMs() const {
        return timer.getMilliseconds();
    }
};

// Global benchmark stats instance
extern BenchmarkStats g_benchmark_stats;

// Macros for easy benchmarking
#define BENCHMARK_SCOPE(name) ScopedTimer __scoped_timer##__LINE__(name, &g_benchmark_stats)
#define BENCHMARK_START(timer_var) Timer timer_var; timer_var.start()
#define BENCHMARK_END(timer_var, name) do { \
    timer_var.stop(); \
    g_benchmark_stats.recordSample(name, timer_var.getMicroseconds()); \
} while(0)

#endif // BENCHMARK_H
