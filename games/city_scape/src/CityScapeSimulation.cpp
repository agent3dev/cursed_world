#include "../include/CityScapeSimulation.h"
#include "../include/CityTerrain.h"
#include "../include/Citizen.h"
#include "../include/Vehicle.h"
#include "../../../common/include/TerminalMatrix.h"
#include "../../../common/include/Tile.h"
#include <ncurses.h>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

CityScapeSimulation::CityScapeSimulation()
    : Simulation("City Scape - Urban Navigation Simulation", 150),  // 150ms frame delay
      citizenCount(0), vehicleCount(0) {
}

void CityScapeSimulation::initialize() {
    // Call base class initialization (sets up matrix, calls initializeTerrain)
    Simulation::initialize();

    // No additional game-specific initialization needed
    // (terrain initialization is done in initializeTerrain())
}

void CityScapeSimulation::initializeTerrain() {
    if (!matrix) return;

    std::random_device rd;
    std::mt19937 gen(rd());

    CityConfig config = CityConfig::getDefaultConfig();

    // First, fill everything with empty space (streets will be empty)
    for (int y = 1; y < matrix->getHeight() - 1; y++) {
        for (int x = 0; x < matrix->getWidth(); x++) {
            Tile* tile = matrix->getTile(x, y);
            if (tile) {
                tile->setChar("  ");  // Empty space for streets
                tile->setWalkable(true);
            }
        }
    }

    // Calculate map center for district division
    int centerX = matrix->getWidth() / 2;
    int centerY = matrix->getHeight() / 2;

    // Randomize which district goes in which quadrant
    std::vector<DistrictType> districtQuadrants = {
        DistrictType::SUBURB,
        DistrictType::BUSINESS,
        DistrictType::INDUSTRIAL,
        DistrictType::CULTURAL
    };
    std::shuffle(districtQuadrants.begin(), districtQuadrants.end(), gen);

    // Generate city blocks - leave 1-tile street inside border
    std::vector<CityBlock> blocks;
    int currentY = 2;  // Start with 1-tile gap after top border (inner street)

    // Generate blocks in rows, leaving space for inner street along borders
    while (currentY < matrix->getHeight() - 3) {  // Leave space for inner street before bottom border
        int currentX = 2;  // Start with 1-tile gap after left border (inner street)
        int maxHeightInRow = 0;  // Track tallest block in this row

        // Generate a row of blocks
        while (currentX < matrix->getWidth() - 3) {  // Leave space for inner street before right border
            // Determine which district this position is in
            DistrictType districtType;
            if (currentX < centerX && currentY < centerY) {
                districtType = districtQuadrants[0];
            } else if (currentX >= centerX && currentY < centerY) {
                districtType = districtQuadrants[1];
            } else if (currentX < centerX && currentY >= centerY) {
                districtType = districtQuadrants[2];
            } else {
                districtType = districtQuadrants[3];
            }

            // Get the district's block sizes
            District district;
            switch (districtType) {
                case DistrictType::SUBURB:
                    district = District::getSuburb();
                    break;
                case DistrictType::BUSINESS:
                    district = District::getBusiness();
                    break;
                case DistrictType::INDUSTRIAL:
                    district = District::getIndustrial();
                    break;
                case DistrictType::CULTURAL:
                    district = District::getCultural();
                    break;
            }

            // Select random block size from district's available sizes
            std::uniform_int_distribution<> block_size_dist(0, district.blockSizes.size() - 1);
            auto [blockWidth, blockHeight] = district.blockSizes[block_size_dist(gen)];

            // Check if block fits (with 1-tile street spacing)
            if (currentX + blockWidth + config.streetWidth <= matrix->getWidth() - 2 &&
                currentY + blockHeight + config.streetWidth <= matrix->getHeight() - 2) {

                // Create block
                CityBlock block(currentX, currentY, blockWidth, blockHeight);

                // Select position for special building within block
                std::uniform_int_distribution<> bx_dist(0, blockWidth - 1);
                std::uniform_int_distribution<> by_dist(0, blockHeight - 1);
                block.specialBuildingX = bx_dist(gen);
                block.specialBuildingY = by_dist(gen);

                blocks.push_back(block);

                // Track the tallest block in this row
                if (blockHeight > maxHeightInRow) {
                    maxHeightInRow = blockHeight;
                }

                // Move to next block position (add block width + 1-tile street)
                currentX += blockWidth + config.streetWidth;
            } else {
                break;  // No more room in this row
            }
        }

        // Move to next row (add tallest block height + 1-tile street)
        if (maxHeightInRow > 0) {
            currentY += maxHeightInRow + config.streetWidth;
        } else {
            break;  // No blocks were placed, stop
        }
    }

    // Now fill each block with buildings based on district
    std::uniform_int_distribution<> park_dist(0, 2);
    const char* parks[] = {"🌳", "🌲", "🌴"};

    for (const auto& block : blocks) {
        // Determine which district this block is in (quadrant-based, randomized)
        DistrictType districtType;
        if (block.x < centerX && block.y < centerY) {
            // Top-left quadrant
            districtType = districtQuadrants[0];
        } else if (block.x >= centerX && block.y < centerY) {
            // Top-right quadrant
            districtType = districtQuadrants[1];
        } else if (block.x < centerX && block.y >= centerY) {
            // Bottom-left quadrant
            districtType = districtQuadrants[2];
        } else {
            // Bottom-right quadrant
            districtType = districtQuadrants[3];
        }

        // Get the district based on type
        District district;
        switch (districtType) {
            case DistrictType::SUBURB:
                district = District::getSuburb();
                break;
            case DistrictType::BUSINESS:
                district = District::getBusiness();
                break;
            case DistrictType::INDUSTRIAL:
                district = District::getIndustrial();
                break;
            case DistrictType::CULTURAL:
                district = District::getCultural();
                break;
        }

        // Select buildings from the district's lists
        std::uniform_int_distribution<> district_regular_dist(0, district.regularBuildings.size() - 1);
        std::uniform_int_distribution<> district_special_dist(0, district.specialBuildings.size() - 1);

        // Select special building emoji from district
        const char* specialBuildingEmoji = district.specialBuildings[district_special_dist(gen)].c_str();

        // Fill the block
        for (int by = 0; by < block.height; by++) {
            for (int bx = 0; bx < block.width; bx++) {
                int worldX = block.x + bx;
                int worldY = block.y + by;

                Tile* tile = matrix->getTile(worldX, worldY);
                if (!tile) continue;

                // Check if this is the special building position
                if (bx == block.specialBuildingX && by == block.specialBuildingY) {
                    // Place special building
                    tile->setChar(specialBuildingEmoji);
                    tile->setWalkable(false);
                } else {
                    // 20% chance for park, otherwise regular building from district
                    if (park_dist(gen) == 0) {  // ~33% chance
                        std::uniform_int_distribution<> park_type(0, 2);
                        tile->setChar(parks[park_type(gen)]);
                        tile->setWalkable(false);  // Parks are obstacles
                    } else {
                        // Place regular building from district
                        const char* regularBuildingEmoji = district.regularBuildings[district_regular_dist(gen)].c_str();
                        tile->setChar(regularBuildingEmoji);
                        tile->setWalkable(false);
                    }
                }
            }
        }
    }

    // Draw border
    matrix->margin("🧱");

    // Spawn vehicles only (no pedestrians)
    std::uniform_int_distribution<> pos_x_dist(1, matrix->getWidth() - 2);
    std::uniform_int_distribution<> pos_y_dist(1, matrix->getHeight() - 2);

    // Spawn 15 vehicles of random types
    std::uniform_int_distribution<> vehicle_type_dist(0, 10);  // 11 vehicle types (0-10)

    for (int i = 0; i < 15; i++) {
        int x = pos_x_dist(gen);
        int y = pos_y_dist(gen);

        // Random direction
        Direction dir = static_cast<Direction>(gen() % 4);

        // Random vehicle type
        VehicleType vehicleType = static_cast<VehicleType>(vehicle_type_dist(gen));

        Vehicle* vehicle = new Vehicle(x, y, vehicleType, dir);

        Tile* tile = matrix->getTile(x, y);
        if (tile && tile->isWalkable()) {
            tile->setActuator(vehicle);
            vehicleCount++;
        } else {
            delete vehicle;
        }
    }
}

void CityScapeSimulation::updateEntities() {
    if (!matrix) return;

    // Update all tiles (which will update their actuators)
    for (int y = 0; y < matrix->getHeight(); y++) {
        for (int x = 0; x < matrix->getWidth(); x++) {
            Tile* tile = matrix->getTile(x, y);
            if (tile && tile->hasActuator()) {
                tile->getActuator()->update(*matrix);
            }
        }
    }
}

void CityScapeSimulation::handleGameInput(int ch) {
    // No game-specific input handling needed for city scape yet
    // (Common inputs like pause, type toggle, quit are handled by base class)
}

void CityScapeSimulation::renderStats() {
    if (!matrix) return;

    std::stringstream ss;
    ss << "Tick: " << tickCount  // Use base class tickCount
       << " | Citizens: " << citizenCount
       << " | Vehicles: " << vehicleCount
       << " | " << (paused ? "[PAUSED]" : "[RUNNING]");  // Use base class paused

    matrix->setDashboard(ss.str());
}
