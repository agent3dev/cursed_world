#include "../include/CityTerrain.h"

CityConfig CityConfig::getDefaultConfig() {
    CityConfig config;
    config.streetWidth = 1;      // 1 tile wide streets between blocks
    config.sidewalkWidth = 0;    // No sidewalks
    return config;
}
