#ifndef TERRAIN_TYPES_H
#define TERRAIN_TYPES_H

#include <string>
#include <vector>
#include <map>

// Generic terrain types for all simulations
// Each simulation can use its own subset or extend this
enum class TerrainType {
    EMPTY,
    PLANTS,
    SEEDLINGS,
    DEAD_TREES,
    ROCKS,
    SEED  // Seed that will grow into seedling
};

// Unicode character options for each terrain type (evolution-specific)
const std::map<TerrainType, std::vector<std::string>> TERRAIN_CHARS = {
    {TerrainType::EMPTY, {"  "}},  // Two spaces to match emoji width
    {TerrainType::PLANTS, {"  "}},
    {TerrainType::SEEDLINGS, {"🌱"}},
    {TerrainType::DEAD_TREES, {"🌲"}},  // Using evergreen tree instead
    {TerrainType::ROCKS, {"⬛"}},  // Using black square instead
    {TerrainType::SEED, {"🔸"}}  // Seed
};

#endif
