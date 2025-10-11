#ifndef VEHICLE_H
#define VEHICLE_H

#include "../../../common/include/Actuator.h"
#include <string>
#include <vector>

// Forward declaration
class TerminalMatrix;

// Direction for vehicle movement
enum class Direction {
    NORTH,
    SOUTH,
    EAST,
    WEST,
    STOPPED
};

// Vehicle types with associated properties
enum class VehicleType {
    CAR,           // 🚗
    TAXI,          // 🚕
    SUV,           // 🚙
    BUS,           // 🚌
    AMBULANCE,     // 🚑
    FIRE_TRUCK,    // 🚒
    POLICE_CAR,    // 🚓
    TRUCK,         // 🚚
    MOTORCYCLE,    // 🏍
    BICYCLE,       // 🚲
    SCOOTER        // 🛵
};

// Vehicle properties structure
struct VehicleProperties {
    std::string emoji;
    int weight;        // Weight in arbitrary units (heavier = slower)
    double speed;      // Speed (inversely proportional to weight)

    VehicleProperties(const char* e, int w)
        : emoji(e), weight(w), speed(100.0 / w) {}
};

// Tile information structure for vehicle sensors
struct TileInfo {
    bool isWalkable;        // Can the vehicle move here?
    bool hasObstacle;       // Is there a building/water?
    bool hasVehicle;        // Is there another vehicle?
    bool hasCitizen;        // Is there a pedestrian?
    bool isRoad;            // Is it a road tile?
    bool isSidewalk;        // Is it a sidewalk?
    bool isIntersection;    // Is it an intersection?
    bool isOutOfBounds;     // Is it outside the matrix?

    TileInfo() : isWalkable(false), hasObstacle(true), hasVehicle(false),
                 hasCitizen(false), isRoad(false), isSidewalk(false),
                 isIntersection(false), isOutOfBounds(true) {}
};

// Vehicle: Cars, buses, etc. navigating roads
class Vehicle : public Actuator {
private:
    VehicleType type;
    VehicleProperties properties;
    Direction currentDirection;
    bool stopped;          // At intersection/traffic
    int stopCounter;       // How long to stay stopped
    int moveCounter;       // For speed-based movement timing

    // Movement logic
    void moveInDirection(TerminalMatrix& matrix);
    bool canMoveForward(TerminalMatrix& matrix);
    void handleIntersection(TerminalMatrix& matrix);

    // Sensor system - get information about surrounding tiles
    TileInfo getTileInfo(TerminalMatrix& matrix, int dx, int dy);
    std::vector<TileInfo> getSurroundingTiles(TerminalMatrix& matrix);

    // Get properties for a vehicle type
    static VehicleProperties getPropertiesForType(VehicleType type);

public:
    Vehicle(int startX, int startY, VehicleType vehicleType, Direction dir);
    ~Vehicle() override;

    void update(TerminalMatrix& matrix) override;
    void stop(int duration);
    Direction getDirection() const { return currentDirection; }
    void setDirection(Direction dir) { currentDirection = dir; }
    VehicleType getType() const { return type; }
    int getWeight() const { return properties.weight; }
    double getSpeed() const { return properties.speed; }
};

#endif
