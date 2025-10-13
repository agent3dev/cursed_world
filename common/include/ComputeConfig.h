#ifndef COMPUTE_CONFIG_H
#define COMPUTE_CONFIG_H

#include "ComputeBackend.h"
#include <string>
#include <map>

/**
 * Configuration for compute backend selection
 *
 * This class handles:
 * - Loading configuration from file (YAML/INI format)
 * - Parsing command-line arguments
 * - Automatic backend selection based on population size
 * - Fallback strategies when preferred backend unavailable
 */
class ComputeConfig {
private:
    BackendType preferred_backend_;
    bool fallback_to_cpu_;
    int auto_threshold_;  // Population size threshold for AUTO mode

    // Configuration file path
    std::string config_file_path_;

    // Load configuration from file
    bool loadFromFile(const std::string& filepath);

public:
    ComputeConfig();

    /**
     * Parse command-line arguments
     *
     * Supported arguments:
     * --backend=cpu|cuda|auto
     * --no-fallback
     * --config=path/to/config.yaml
     *
     * @param argc - Argument count
     * @param argv - Argument vector
     * @return True if parsing succeeded, false if invalid arguments
     */
    bool parseCommandLine(int argc, char* argv[]);

    /**
     * Load configuration from file
     *
     * @param filepath - Path to configuration file
     * @return True if file loaded successfully
     */
    bool loadConfig(const std::string& filepath);

    /**
     * Save current configuration to file
     *
     * @param filepath - Path to save configuration
     * @return True if saved successfully
     */
    bool saveConfig(const std::string& filepath) const;

    /**
     * Select backend based on configuration and population size
     *
     * @param population_size - Current population size
     * @return Backend type to use
     */
    BackendType selectBackend(int population_size) const;

    /**
     * Get preferred backend type
     */
    BackendType getPreferredBackend() const { return preferred_backend_; }

    /**
     * Set preferred backend type
     */
    void setPreferredBackend(BackendType type) { preferred_backend_ = type; }

    /**
     * Check if CPU fallback is enabled
     */
    bool isFallbackEnabled() const { return fallback_to_cpu_; }

    /**
     * Enable/disable CPU fallback
     */
    void setFallbackEnabled(bool enabled) { fallback_to_cpu_ = enabled; }

    /**
     * Get population threshold for AUTO mode
     */
    int getAutoThreshold() const { return auto_threshold_; }

    /**
     * Set population threshold for AUTO mode
     *
     * @param threshold - Switch to GPU when population >= this value
     */
    void setAutoThreshold(int threshold) { auto_threshold_ = threshold; }

    /**
     * Print current configuration
     */
    void print() const;

    /**
     * Get default configuration file path
     */
    static std::string getDefaultConfigPath();
};

/**
 * Helper function to parse --backend argument value
 *
 * Examples:
 * --backend=cpu    -> BackendType::CPU
 * --backend=cuda   -> BackendType::CUDA
 * --backend=auto   -> BackendType::AUTO
 *
 * @param value - String value to parse
 * @return Parsed backend type, or AUTO if invalid
 */
BackendType parseBackendArgument(const std::string& value);

/**
 * Helper function to extract key-value from argument
 *
 * Examples:
 * "--backend=cpu" -> {"backend", "cpu"}
 * "--config=path" -> {"config", "path"}
 *
 * @param arg - Command-line argument string
 * @param key - Output: key part
 * @param value - Output: value part
 * @return True if argument is in key=value format
 */
bool parseKeyValue(const std::string& arg, std::string& key, std::string& value);

#endif // COMPUTE_CONFIG_H
