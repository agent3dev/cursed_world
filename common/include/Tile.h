#ifndef TILE_H
#define TILE_H

#include <string>
#include <vector>
#include <map>
#include "Actuator.h"
#include "TerrainTypes.h"  // Common terrain types

enum class TileType {
    TERRAIN,
    SOIL
};

enum class SoilType {
    DRY,
    WET,
    FERTILE,
    BARREN,
    ROCKY,
    SANDY
};

class Tile {
private:
    std::string displayChar;  // Store as string to avoid dangling pointers
    bool walkable;
    bool transparent;
    bool edible;  // Can this tile be eaten as food?
    int colorPair;
    TileType type;
    TerrainType terrainType;  // From EvolutionTerrain.h - could be made generic in future
    SoilType soilType;
    Actuator* actuator;  // nullptr if no actuator on this tile
    int growthTimer;  // Ticks until seed grows into seedling (0 = not growing)

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
    int getGrowthTimer() const { return growthTimer; }

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
    void setGrowthTimer(int ticks) { growthTimer = ticks; }

    // Growth logic
    void tickGrowth();  // Decrease growth timer and convert to seedling when ready
};

#endif
