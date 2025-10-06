#include "Tile.h"

Tile::Tile(const std::string& c, bool walk, bool trans, int color, TileType t, TerrainType tt, SoilType st)
    : displayChar(c), walkable(walk), transparent(trans), edible(false), colorPair(color), type(t), terrainType(tt), soilType(st), actuator(nullptr), growthTimer(0) {
}

Tile::~Tile() {
    // Note: Actuator deletion managed externally
    // We don't delete actuator here to avoid double-deletion
}

void Tile::tickGrowth() {
    if (growthTimer > 0 && terrainType == TerrainType::SEED) {
        growthTimer--;
        if (growthTimer == 0) {
            // Grow into seedling
            terrainType = TerrainType::SEEDLINGS;
            displayChar = "🌱";
            edible = true;
        }
    }
}
