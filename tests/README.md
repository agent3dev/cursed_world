# Cursed World Test Suite

Comprehensive unit tests for the Cursed World simulation framework using Google Test.

## Test Coverage

### Common Framework Tests (18 tests)

#### Tile Tests (8 tests)
- `test_tile.cpp` - Tests for Tile class
  - Default construction and properties
  - Character, walkable, edible properties
  - Terrain type management
  - Actuator management
  - Growth timer and seed→seedling conversion
  - Combined property handling

#### Actuator Tests (8 tests)
- `test_actuator.cpp` - Tests for Actuator base class
  - Default and parameterized construction
  - Position setters (X, Y, setPosition)
  - Character and type management
  - Blocking property
  - Color pair support
  - All actuator types (CHARACTER, NPC, TRAP, ITEM, ENEMY, RODENT, CAT, CITIZEN, VEHICLE)

#### TerminalMatrix Tests (10 tests)
- `test_terminal_matrix.cpp` - Tests for grid rendering engine
  - Construction with dashboard support
  - Tile access and bounds checking
  - Character setting
  - Type view flag
  - Dashboard text management
  - Window title setting
  - Growth timer update system
  - Wall animation state
  - Grid integrity across all tiles

### Evolution Game Tests (12 tests)

#### NeuralNetwork Tests (12 tests)
- `test_neural_network.cpp` - Tests for recurrent neural network
  - Construction and architecture
  - Forward pass dimensions and values
  - Recurrent memory (hidden state persistence)
  - Hidden state reset
  - Weight access and management
  - Weight mutation (with and without rate)
  - Network copying
  - Output range validation (tanh: -1 to 1)
  - Mouse brain architecture (9→16→9)
  - Cat brain architecture (10→16→9)

## Running Tests

### Build and Run All Tests
```bash
cd tests
make
./run_tests
```

### Run from Project Root
```bash
make test
```

### Clean Test Build
```bash
cd tests
make clean
```

## Test Results

**Current Status: ✅ 38/38 tests passing (100%)**

```
[==========] Running 38 tests from 4 test suites.
[  PASSED  ] 38 tests.
```

## Dependencies

- **Google Test 1.17.0** - C++ testing framework
- **g++ with C++17** - Compiler
- **ncurses** - Terminal UI library (for TerminalMatrix)
- **yaml-cpp** - Configuration (for NeuralNetwork file I/O)

Install on Arch Linux:
```bash
sudo pacman -S gtest
```

Install on Ubuntu/Debian:
```bash
sudo apt install libgtest-dev
```

## Writing New Tests

### Test Structure

```cpp
#include <gtest/gtest.h>
#include "YourClass.h"

class YourClassTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup before each test
    }

    void TearDown() override {
        // Cleanup after each test
    }
};

TEST_F(YourClassTest, TestName) {
    // Arrange
    YourClass obj;

    // Act
    obj.doSomething();

    // Assert
    EXPECT_EQ(obj.getValue(), expected);
}
```

### Adding Tests to Build

Edit `tests/Makefile` and add your test file:

```makefile
TEST_SOURCES = test_tile.cpp \
               test_actuator.cpp \
               test_your_new_test.cpp
```

## Test Philosophy

1. **Unit Tests Only** - Test individual components in isolation
2. **No ncurses Rendering** - Tests check logic without actual terminal rendering
3. **Fast Execution** - All tests run in < 1 second
4. **Deterministic** - Tests produce consistent results
5. **Comprehensive** - Cover normal cases, edge cases, and error conditions

## Coverage Analysis

| Component | Lines | Tests | Coverage |
|-----------|-------|-------|----------|
| Tile | 23 | 8 | ✅ High |
| Actuator | 20 | 8 | ✅ High |
| TerminalMatrix | 165 | 10 | 🟡 Medium |
| NeuralNetwork | 150 | 12 | ✅ High |

**Notes:**
- TerminalMatrix rendering logic not fully tested (requires ncurses mocking)
- PopulationManager tests not yet implemented
- City Scape pathfinding tests not yet implemented

## Future Test Additions

- [ ] PopulationManager tests (selection, mutation, evolution)
- [ ] Citizen pathfinding tests
- [ ] Vehicle navigation tests
- [ ] Border animation tests
- [ ] Integration tests (multi-component)
- [ ] Performance benchmarks

## Continuous Integration

To run tests before commits:

```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
cd tests && make clean && make && ./run_tests
if [ $? -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi
```

## Debugging Failed Tests

### Run Specific Test Suite
```bash
./run_tests --gtest_filter=TileTest.*
```

### Run Single Test
```bash
./run_tests --gtest_filter=TileTest.GrowthTimer
```

### Verbose Output
```bash
./run_tests --gtest_verbose
```

### List All Tests
```bash
./run_tests --gtest_list_tests
```

---

**Test suite maintained as part of Cursed World refactoring initiative.**
