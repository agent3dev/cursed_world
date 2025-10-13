#include <gtest/gtest.h>
#include "Actuator.h"

// Test fixture for Actuator tests
class ActuatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup runs before each test
    }

    void TearDown() override {
        // Cleanup runs after each test
    }
};

// Test default construction
TEST_F(ActuatorTest, DefaultConstruction) {
    Actuator actor;

    EXPECT_EQ(actor.getX(), 0);
    EXPECT_EQ(actor.getY(), 0);
    EXPECT_EQ(actor.getChar(), "@");
    EXPECT_EQ(actor.getType(), ActuatorType::CHARACTER);
    EXPECT_TRUE(actor.isBlocking());
    EXPECT_EQ(actor.getColorPair(), 0);
}

// Test parameterized construction
TEST_F(ActuatorTest, ParameterizedConstruction) {
    Actuator actor(10, 20, "🐀", ActuatorType::RODENT, false, 5);

    EXPECT_EQ(actor.getX(), 10);
    EXPECT_EQ(actor.getY(), 20);
    EXPECT_EQ(actor.getChar(), "🐀");
    EXPECT_EQ(actor.getType(), ActuatorType::RODENT);
    EXPECT_FALSE(actor.isBlocking());
    EXPECT_EQ(actor.getColorPair(), 5);
}

// Test position setters
TEST_F(ActuatorTest, PositionSetters) {
    Actuator actor;

    actor.setX(15);
    EXPECT_EQ(actor.getX(), 15);

    actor.setY(25);
    EXPECT_EQ(actor.getY(), 25);

    actor.setPosition(30, 40);
    EXPECT_EQ(actor.getX(), 30);
    EXPECT_EQ(actor.getY(), 40);
}

// Test character setter
TEST_F(ActuatorTest, CharacterSetter) {
    Actuator actor;

    actor.setChar("X");
    EXPECT_EQ(actor.getChar(), "X");

    actor.setChar("🐈");
    EXPECT_EQ(actor.getChar(), "🐈");
}

// Test type setter
TEST_F(ActuatorTest, TypeSetter) {
    Actuator actor;

    EXPECT_EQ(actor.getType(), ActuatorType::CHARACTER);

    actor.setType(ActuatorType::CAT);
    EXPECT_EQ(actor.getType(), ActuatorType::CAT);

    actor.setType(ActuatorType::CITIZEN);
    EXPECT_EQ(actor.getType(), ActuatorType::CITIZEN);
}

// Test blocking property
TEST_F(ActuatorTest, BlockingProperty) {
    Actuator actor;

    EXPECT_TRUE(actor.isBlocking());

    actor.setBlocking(false);
    EXPECT_FALSE(actor.isBlocking());

    actor.setBlocking(true);
    EXPECT_TRUE(actor.isBlocking());
}

// Test color pair
TEST_F(ActuatorTest, ColorPairSetter) {
    Actuator actor;

    EXPECT_EQ(actor.getColorPair(), 0);

    actor.setColorPair(3);
    EXPECT_EQ(actor.getColorPair(), 3);

    actor.setColorPair(10);
    EXPECT_EQ(actor.getColorPair(), 10);
}

// Test all actuator types
TEST_F(ActuatorTest, AllActuatorTypes) {
    Actuator character(0, 0, "@", ActuatorType::CHARACTER);
    EXPECT_EQ(character.getType(), ActuatorType::CHARACTER);

    Actuator npc(0, 0, "N", ActuatorType::NPC);
    EXPECT_EQ(npc.getType(), ActuatorType::NPC);

    Actuator trap(0, 0, "^", ActuatorType::TRAP);
    EXPECT_EQ(trap.getType(), ActuatorType::TRAP);

    Actuator item(0, 0, "i", ActuatorType::ITEM);
    EXPECT_EQ(item.getType(), ActuatorType::ITEM);

    Actuator enemy(0, 0, "E", ActuatorType::ENEMY);
    EXPECT_EQ(enemy.getType(), ActuatorType::ENEMY);

    Actuator rodent(0, 0, "🐀", ActuatorType::RODENT);
    EXPECT_EQ(rodent.getType(), ActuatorType::RODENT);

    Actuator cat(0, 0, "🐈", ActuatorType::CAT);
    EXPECT_EQ(cat.getType(), ActuatorType::CAT);

    Actuator citizen(0, 0, "🚶", ActuatorType::CITIZEN);
    EXPECT_EQ(citizen.getType(), ActuatorType::CITIZEN);

    Actuator vehicle(0, 0, "🚗", ActuatorType::VEHICLE);
    EXPECT_EQ(vehicle.getType(), ActuatorType::VEHICLE);
}
