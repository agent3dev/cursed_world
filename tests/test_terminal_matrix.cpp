#include <gtest/gtest.h>
#include "TerminalMatrix.h"
#include "Tile.h"

// Mock test for TerminalMatrix without initializing ncurses
// We test the data structures and logic, not the rendering

class TerminalMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Don't initialize ncurses for unit tests
    }

    void TearDown() override {
        // No ncurses to clean up
    }
};

// Test construction
TEST_F(TerminalMatrixTest, Construction) {
    TerminalMatrix matrix(10, 20, 1);

    EXPECT_EQ(matrix.getWidth(), 10);
    // Height is reduced by dashboard height (20 - 1 = 19)
    EXPECT_EQ(matrix.getHeight(), 19);
}

// Test tile access
TEST_F(TerminalMatrixTest, TileAccess) {
    TerminalMatrix matrix(5, 5, 0);

    Tile* tile = matrix.getTile(2, 3);
    ASSERT_NE(tile, nullptr);

    // Should be able to modify the tile
    tile->setChar("X");
    EXPECT_EQ(tile->getChar(), "X");

    // Same tile should be retrieved
    Tile* same_tile = matrix.getTile(2, 3);
    EXPECT_EQ(same_tile->getChar(), "X");
}

// Test bounds checking
TEST_F(TerminalMatrixTest, BoundsChecking) {
    TerminalMatrix matrix(10, 10, 0);

    // Valid access
    EXPECT_NE(matrix.getTile(0, 0), nullptr);
    EXPECT_NE(matrix.getTile(9, 9), nullptr);
    EXPECT_NE(matrix.getTile(5, 5), nullptr);

    // Out of bounds should return nullptr
    EXPECT_EQ(matrix.getTile(-1, 0), nullptr);
    EXPECT_EQ(matrix.getTile(0, -1), nullptr);
    EXPECT_EQ(matrix.getTile(10, 0), nullptr);
    EXPECT_EQ(matrix.getTile(0, 10), nullptr);
    EXPECT_EQ(matrix.getTile(100, 100), nullptr);
}

// Test setChar
TEST_F(TerminalMatrixTest, SetChar) {
    TerminalMatrix matrix(5, 5, 0);

    matrix.setChar(2, 2, "🌳");

    Tile* tile = matrix.getTile(2, 2);
    ASSERT_NE(tile, nullptr);
    EXPECT_EQ(tile->getChar(), "🌳");
}

// Test type view flag
TEST_F(TerminalMatrixTest, TypeViewFlag) {
    TerminalMatrix matrix(5, 5, 0);

    EXPECT_FALSE(matrix.getTypeView());

    matrix.setTypeView(true);
    EXPECT_TRUE(matrix.getTypeView());

    matrix.setTypeView(false);
    EXPECT_FALSE(matrix.getTypeView());
}

// Test dashboard text
TEST_F(TerminalMatrixTest, DashboardText) {
    TerminalMatrix matrix(10, 10, 1);

    matrix.setDashboard("Test Dashboard");
    // Dashboard text is set (actual rendering tested separately)
    // Just ensure it doesn't crash
}

// Test window title
TEST_F(TerminalMatrixTest, WindowTitle) {
    TerminalMatrix matrix(10, 10, 0);

    matrix.setWindowTitle("Test Title");
    // Title is set (actual display tested separately)
    // Just ensure it doesn't crash
}

// Test growth timer update
TEST_F(TerminalMatrixTest, GrowthTimerUpdate) {
    TerminalMatrix matrix(5, 5, 0);

    Tile* tile = matrix.getTile(2, 2);
    ASSERT_NE(tile, nullptr);

    // Set up tile as a SEED to enable growth
    tile->setGrowthTimer(10);
    tile->setTerrainType(TerrainType::SEED);

    // Update growth decrements SEED timers
    matrix.updateGrowth();

    // Growth timer should have decremented
    EXPECT_EQ(tile->getGrowthTimer(), 9);
}

// Test wall animation state
TEST_F(TerminalMatrixTest, WallAnimationState) {
    TerminalMatrix matrix(10, 10, 0);

    // Toggle wall animation (implementation detail, just ensure no crash)
    matrix.toggleWallAnimation();
    matrix.toggleWallAnimation();
}

// Test grid integrity
TEST_F(TerminalMatrixTest, GridIntegrity) {
    TerminalMatrix matrix(8, 8, 0);

    // Set different characters in a pattern
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            std::string marker = std::to_string(x) + "," + std::to_string(y);
            matrix.setChar(x, y, marker);
        }
    }

    // Verify all tiles have correct data
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            Tile* tile = matrix.getTile(x, y);
            ASSERT_NE(tile, nullptr);

            std::string expected = std::to_string(x) + "," + std::to_string(y);
            EXPECT_EQ(tile->getChar(), expected);
        }
    }
}
