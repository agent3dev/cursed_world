#ifndef CITY_TERRAIN_H
#define CITY_TERRAIN_H

#include <string>
#include <vector>
#include <map>

// Urban terrain types
enum class UrbanTerrain {
    ROAD,              // Roads for vehicles
    SIDEWALK,          // Sidewalks for pedestrians
    REGULAR_BUILDING,  // Common buildings (not walkable)
    SPECIAL_BUILDING,  // Landmark/unique buildings (not walkable)
    PARK,              // Parks and green spaces
    WATER,             // Fountains, rivers
    PARKING,           // Parking lots
    CROSSWALK,         // Pedestrian crossings
    INTERSECTION       // Road intersections
};

// Regular buildings - common/everyday structures
static const std::vector<std::string> REGULAR_BUILDINGS = {
    "🏢",  // Office building
    "🏬",  // Department store
    "🏦",  // Bank
    "🏨",  // Hotel
    "🏪",  // Convenience store
    "🏫",  // School
    "🏥",  // Hospital
    "🏭",  // Factory
    "🏠",  // House
    "🏡",  // House with garden
};

// Special buildings - landmarks and unique structures
static const std::vector<std::string> SPECIAL_BUILDINGS = {
    "⛪",  // Church
    "🕌",  // Mosque
    "🕍",  // Synagogue
};

// District types
enum class DistrictType {
    SUBURB,         // Residential: houses, parks, churches, stores
    BUSINESS,       // Downtown: offices, banks, hotels, museums
    INDUSTRIAL,     // Factories, construction, abandoned buildings
    CULTURAL        // Schools, hospitals, museums, parks
};

// District building definitions
struct District {
    std::vector<std::string> regularBuildings;
    std::vector<std::string> specialBuildings;
    std::vector<std::pair<int, int>> blockSizes;

    static District getSuburb() {
        return {
            {"🏠", "🏡", "🏪"},  // Houses and convenience stores
            {"⛪", "🕌", "🕍"},   // Religious buildings
            {{3, 3}}  // Uniform 3x3 blocks
        };
    }

    static District getBusiness() {
        return {
            {"🏢", "🏦", "🏨", "🏬"},  // Offices, banks, hotels, stores
            {"⛪", "🕌", "🕍"},         // Religious buildings (shared)
            {{3, 3}}  // Uniform 3x3 blocks
        };
    }

    static District getIndustrial() {
        return {
            {"🏭", "🏪"},        // Factories, stores for workers
            {"⛪", "🕌", "🕍"},   // Religious buildings (shared)
            {{3, 3}}  // Uniform 3x3 blocks
        };
    }

    static District getCultural() {
        return {
            {"🏫", "🏥"},        // Schools, hospitals
            {"⛪", "🕌", "🕍"},   // Religious buildings (shared)
            {{3, 3}}  // Uniform 3x3 blocks
        };
    }
};

// Emoji representations for each terrain type (parking, crosswalk, intersection not used)
static const std::map<UrbanTerrain, std::vector<std::string>> URBAN_TERRAIN_CHARS = {
    {UrbanTerrain::ROAD,              {"⬛"}},
    {UrbanTerrain::SIDEWALK,          {"⬜"}},
    {UrbanTerrain::REGULAR_BUILDING,  REGULAR_BUILDINGS},
    {UrbanTerrain::SPECIAL_BUILDING,  SPECIAL_BUILDINGS},
    {UrbanTerrain::PARK,              {"🌳", "🌲", "🌴"}},
    {UrbanTerrain::WATER,             {"💧", "🌊"}}
};

// City block structure
struct CityBlock {
    int x, y;           // Top-left corner position
    int width, height;  // Block dimensions
    int specialBuildingX, specialBuildingY;  // Position of the special building within block

    CityBlock(int posX, int posY, int w, int h)
        : x(posX), y(posY), width(w), height(h),
          specialBuildingX(-1), specialBuildingY(-1) {}
};

// Available block sizes
static const std::vector<std::pair<int, int>> BLOCK_SIZES = {
    {1, 3},  // Thin vertical
    {3, 1},  // Thin horizontal
    {2, 2},  // Small square
    {3, 2},  // Horizontal rectangle
    {2, 3},  // Vertical rectangle
    {3, 3},  // Medium square
    {4, 2},  // Wide rectangle
    {2, 4},  // Tall rectangle
    {4, 3},  // Large horizontal
    {3, 4},  // Large vertical
};

// Configuration for terrain generation
struct CityConfig {
    int streetWidth;        // Width of streets between blocks (1-2 tiles)
    int sidewalkWidth;      // Width of sidewalk around blocks (0-1 tiles)

    static CityConfig getDefaultConfig();
};

#endif
