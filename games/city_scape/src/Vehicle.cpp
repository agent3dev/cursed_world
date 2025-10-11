#include "../include/Vehicle.h"
#include "../../../common/include/TerminalMatrix.h"
#include "../../../common/include/Tile.h"

// Static method to get properties for each vehicle type
VehicleProperties Vehicle::getPropertiesForType(VehicleType type) {
    switch (type) {
        case VehicleType::BICYCLE:     return VehicleProperties("🚲", 10);   // Lightest, fastest
        case VehicleType::SCOOTER:     return VehicleProperties("🛵", 15);
        case VehicleType::MOTORCYCLE:  return VehicleProperties("🏍", 20);
        case VehicleType::CAR:         return VehicleProperties("🚗", 30);
        case VehicleType::TAXI:        return VehicleProperties("🚕", 30);
        case VehicleType::POLICE_CAR:  return VehicleProperties("🚓", 32);
        case VehicleType::SUV:         return VehicleProperties("🚙", 40);
        case VehicleType::AMBULANCE:   return VehicleProperties("🚑", 45);
        case VehicleType::TRUCK:       return VehicleProperties("🚚", 60);
        case VehicleType::FIRE_TRUCK:  return VehicleProperties("🚒", 70);
        case VehicleType::BUS:         return VehicleProperties("🚌", 80);   // Heaviest, slowest
        default:                       return VehicleProperties("🚗", 30);
    }
}

Vehicle::Vehicle(int startX, int startY, VehicleType vehicleType, Direction dir)
    : Actuator(startX, startY, "", ActuatorType::VEHICLE),
      type(vehicleType),
      properties(getPropertiesForType(vehicleType)),
      currentDirection(dir),
      stopped(false),
      stopCounter(0),
      moveCounter(0) {
    // Set the display character to the vehicle's emoji
    displayChar = properties.emoji;
}

Vehicle::~Vehicle() {
}

void Vehicle::moveInDirection(TerminalMatrix& matrix) {
    if (stopped) {
        stopCounter--;
        if (stopCounter <= 0) {
            stopped = false;
        }
        return;
    }

    int moveX = 0;
    int moveY = 0;

    switch (currentDirection) {
        case Direction::NORTH:
            moveY = -1;
            break;
        case Direction::SOUTH:
            moveY = 1;
            break;
        case Direction::EAST:
            moveX = 1;
            break;
        case Direction::WEST:
            moveX = -1;
            break;
        case Direction::STOPPED:
            return;
    }

    if (canMoveForward(matrix)) {
        move(moveX, moveY, matrix);
    } else {
        // Hit obstacle or edge, stop or turn around
        handleIntersection(matrix);
    }
}

bool Vehicle::canMoveForward(TerminalMatrix& matrix) {
    int checkX = x;
    int checkY = y;

    switch (currentDirection) {
        case Direction::NORTH: checkY--; break;
        case Direction::SOUTH: checkY++; break;
        case Direction::EAST:  checkX++; break;
        case Direction::WEST:  checkX--; break;
        case Direction::STOPPED: return false;
    }

    Tile* tile = matrix.getTile(checkX, checkY);
    if (!tile) return false;
    if (!tile->isWalkable()) return false;
    if (tile->hasActuator()) return false;
    return true;
}

void Vehicle::handleIntersection(TerminalMatrix& matrix) {
    // Simple logic: try to turn, or stop
    // Try turning right first
    Direction newDir = currentDirection;

    switch (currentDirection) {
        case Direction::NORTH: newDir = Direction::EAST; break;
        case Direction::EAST:  newDir = Direction::SOUTH; break;
        case Direction::SOUTH: newDir = Direction::WEST; break;
        case Direction::WEST:  newDir = Direction::NORTH; break;
        default: break;
    }

    currentDirection = newDir;
    stop(3);  // Stop for 3 ticks at intersection
}

void Vehicle::stop(int duration) {
    stopped = true;
    stopCounter = duration;
}

// Get information about a specific tile relative to vehicle position
TileInfo Vehicle::getTileInfo(TerminalMatrix& matrix, int dx, int dy) {
    TileInfo info;

    Tile* tile = matrix.getTile(x + dx, y + dy);

    // Check if tile exists
    if (!tile) {
        info.isOutOfBounds = true;
        return info;
    }

    info.isOutOfBounds = false;
    info.isWalkable = tile->isWalkable();
    info.hasObstacle = !tile->isWalkable();

    // Check for actuators
    if (tile->hasActuator()) {
        Actuator* act = tile->getActuator();
        if (act) {
            ActuatorType actType = act->getType();
            if (actType == ActuatorType::VEHICLE) {
                info.hasVehicle = true;
            } else if (actType == ActuatorType::CITIZEN) {
                info.hasCitizen = true;
            }
        }
    }

    // Identify terrain by character
    std::string charStr = tile->getChar();

    // Empty space is road/street
    if (charStr == "  ") {
        info.isRoad = true;
    }
    // Parks (now obstacles)
    else if (charStr == "🌳" || charStr == "🌲" || charStr == "🌴") {
        // Parks are obstacles now
        info.isRoad = false;
        info.hasObstacle = true;
    }

    return info;
}

// Get information about all 8 surrounding tiles
// Order: N, NE, E, SE, S, SW, W, NW
std::vector<TileInfo> Vehicle::getSurroundingTiles(TerminalMatrix& matrix) {
    std::vector<TileInfo> surroundings;

    const int dx_order[] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int dy_order[] = {-1, -1, 0, 1, 1, 1, 0, -1};

    for (int i = 0; i < 8; i++) {
        surroundings.push_back(getTileInfo(matrix, dx_order[i], dy_order[i]));
    }

    return surroundings;
}

void Vehicle::update(TerminalMatrix& matrix) {
    // Speed determines how often the vehicle moves
    // Higher speed = moves more frequently
    // Speed is inverse of weight (100/weight)
    // moveCounter increments each update; when it reaches threshold based on speed, vehicle moves

    moveCounter++;

    // Calculate movement threshold: lower speed = higher threshold (moves less often)
    // Speed range: 1.25 (bus, weight 80) to 10.0 (bicycle, weight 10)
    // Convert to movement frequency: move every N ticks
    int movementThreshold = static_cast<int>(10.0 / properties.speed);
    if (movementThreshold < 1) movementThreshold = 1;

    if (moveCounter >= movementThreshold) {
        moveCounter = 0;
        moveInDirection(matrix);
    }
}
