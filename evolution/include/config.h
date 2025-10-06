#ifndef CONFIG_H
#define CONFIG_H

#include <yaml-cpp/yaml.h>
#include <string>

namespace TerrainConfig {
    struct Ratios {
        int empty;
        int seedlings;
        int dead_trees;
        int rocks;
        int total;
    };

    inline Ratios loadConfig(const std::string& filename = "config.yaml") {
        YAML::Node config = YAML::LoadFile(filename);

        Ratios ratios;
        ratios.empty = config["terrain"]["ratios"]["empty"].as<int>();
        ratios.seedlings = config["terrain"]["ratios"]["seedlings"].as<int>();
        ratios.dead_trees = config["terrain"]["ratios"]["dead_trees"].as<int>();
        ratios.rocks = config["terrain"]["ratios"]["rocks"].as<int>();
        ratios.total = ratios.empty + ratios.seedlings + ratios.dead_trees + ratios.rocks;

        return ratios;
    }
}

#endif
