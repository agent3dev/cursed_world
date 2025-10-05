#include "Actuator.h"

Actuator::Actuator(int posX, int posY, const std::string& c, ActuatorType t, bool block, int color)
    : displayChar(c), x(posX), y(posY), type(t), blocking(block), colorPair(color) {
}
