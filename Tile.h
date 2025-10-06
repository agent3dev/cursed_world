#ifndef TILE_H
#define TILE_H

#include <string>
#include <vector>
#include <map>
#include "Actuator.h"

enum class TileType {
    TERRAIN,
    SOIL
};

enum class TerrainType {
    EMPTY,
    PLANTS,
    SEEDLINGS,
    DEAD_TREES,
    ROCKS
};

enum class SoilType {
    DRY,
    WET,
    FERTILE,
    BARREN,
    ROCKY,
    SANDY
};

// Unicode character options for each terrain type
const std::map<TerrainType, std::vector<std::string>> TERRAIN_CHARS = {
    {TerrainType::EMPTY, {"  "}},  // Two spaces to match emoji width
    {TerrainType::PLANTS, {"  "}},
    {TerrainType::SEEDLINGS, {"🌱"}},
    {TerrainType::DEAD_TREES, {"🪾"}},
    {TerrainType::ROCKS, {"🪨"}}
};

class Tile {
private:
    std::string displayChar;
    bool walkable;
    bool transparent;
    bool edible;  // Can this tile be eaten as food?
    int colorPair;
    TileType type;
    TerrainType terrainType;
    SoilType soilType;
    Actuator* actuator;  // nullptr if no actuator on this tile

public:
    Tile(const std::string& c = "  ", bool walk = true, bool trans = true, int color = 0, TileType t = TileType::SOIL, TerrainType tt = TerrainType::EMPTY, SoilType st = SoilType::DRY);
    ~Tile();

    // Getters
    const std::string& getChar() const { return displayChar; }
    bool isWalkable() const { return walkable; }
    bool isTransparent() const { return transparent; }
    bool isEdible() const { return edible; }
    int getColorPair() const { return colorPair; }
    TileType getType() const { return type; }
    TerrainType getTerrainType() const { return terrainType; }
    SoilType getSoilType() const { return soilType; }
    Actuator* getActuator() const { return actuator; }
    bool hasActuator() const { return actuator != nullptr; }

    // Setters
    void setChar(const std::string& c) { displayChar = c; }
    void setWalkable(bool walk) { walkable = walk; }
    void setTransparent(bool trans) { transparent = trans; }
    void setEdible(bool food) { edible = food; }
    void setColorPair(int color) { colorPair = color; }
    void setType(TileType t) { type = t; }
    void setTerrainType(TerrainType tt) { terrainType = tt; }
    void setSoilType(SoilType st) { soilType = st; }
    void setActuator(Actuator* act) { actuator = act; }
};

#endif
