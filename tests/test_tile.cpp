#include <gtest/gtest.h>
#include "Tile.h"
#include "Actuator.h"

// Test fixture for Tile tests
class TileTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup runs before each test
    }

    void TearDown() override {
        // Cleanup runs after each test
    }
};

// Test default construction
TEST_F(TileTest, DefaultConstruction) {
    Tile tile;

    EXPECT_EQ(tile.getChar(), "  ");  // Default empty space
    EXPECT_TRUE(tile.isWalkable());    // Walkable by default
    EXPECT_FALSE(tile.isEdible());     // Not edible by default
    EXPECT_FALSE(tile.hasActuator());  // No actuator by default
}

// Test character setting
TEST_F(TileTest, SetAndGetChar) {
    Tile tile;

    tile.setChar("🌳");
    EXPECT_EQ(tile.getChar(), "🌳");

    tile.setChar("X");
    EXPECT_EQ(tile.getChar(), "X");
}

// Test walkable property
TEST_F(TileTest, WalkableProperty) {
    Tile tile;

    EXPECT_TRUE(tile.isWalkable());

    tile.setWalkable(false);
    EXPECT_FALSE(tile.isWalkable());

    tile.setWalkable(true);
    EXPECT_TRUE(tile.isWalkable());
}

// Test edible property
TEST_F(TileTest, EdibleProperty) {
    Tile tile;

    EXPECT_FALSE(tile.isEdible());

    tile.setEdible(true);
    EXPECT_TRUE(tile.isEdible());

    tile.setEdible(false);
    EXPECT_FALSE(tile.isEdible());
}

// Test terrain type
TEST_F(TileTest, TerrainType) {
    Tile tile;

    tile.setTerrainType(TerrainType::EMPTY);
    EXPECT_EQ(tile.getTerrainType(), TerrainType::EMPTY);

    tile.setTerrainType(TerrainType::SEEDLINGS);
    EXPECT_EQ(tile.getTerrainType(), TerrainType::SEEDLINGS);

    tile.setTerrainType(TerrainType::ROCKS);
    EXPECT_EQ(tile.getTerrainType(), TerrainType::ROCKS);
}

// Test actuator management
TEST_F(TileTest, ActuatorManagement) {
    Tile tile;

    EXPECT_FALSE(tile.hasActuator());
    EXPECT_EQ(tile.getActuator(), nullptr);

    Actuator actor(5, 10, "@", ActuatorType::CHARACTER);
    tile.setActuator(&actor);

    EXPECT_TRUE(tile.hasActuator());
    EXPECT_NE(tile.getActuator(), nullptr);
    EXPECT_EQ(tile.getActuator()->getChar(), "@");

    tile.setActuator(nullptr);
    EXPECT_FALSE(tile.hasActuator());
    EXPECT_EQ(tile.getActuator(), nullptr);
}

// Test growth timer
TEST_F(TileTest, GrowthTimer) {
    Tile tile;

    EXPECT_EQ(tile.getGrowthTimer(), 0);

    tile.setGrowthTimer(100);
    EXPECT_EQ(tile.getGrowthTimer(), 100);

    // Test tickGrowth (only decrements for SEED terrain type)
    tile.setTerrainType(TerrainType::SEED);
    tile.tickGrowth();
    EXPECT_EQ(tile.getGrowthTimer(), 99);

    // When timer reaches 0, should convert to seedling
    tile.setGrowthTimer(1);
    tile.tickGrowth();
    EXPECT_EQ(tile.getGrowthTimer(), 0);
    EXPECT_EQ(tile.getTerrainType(), TerrainType::SEEDLINGS);
    EXPECT_TRUE(tile.isEdible());
}

// Test combined properties
TEST_F(TileTest, CombinedProperties) {
    Tile tile;

    tile.setChar("🌱");
    tile.setWalkable(true);
    tile.setEdible(true);
    tile.setTerrainType(TerrainType::SEEDLINGS);
    tile.setGrowthTimer(50);

    EXPECT_EQ(tile.getChar(), "🌱");
    EXPECT_TRUE(tile.isWalkable());
    EXPECT_TRUE(tile.isEdible());
    EXPECT_EQ(tile.getTerrainType(), TerrainType::SEEDLINGS);
    EXPECT_EQ(tile.getGrowthTimer(), 50);
}
