# Cursed World 🌍

A terminal-based simulation game built with C++ and ncurses, featuring procedurally generated terrain and autonomous AI agents.

## Features

### Terrain System
- **Tile-based world** with multiple terrain types:
  - 🌱 Seedlings (edible)
  - 🪾 Dead Trees (obstacles)
  - 🪨 Rocks (obstacles)
  - Empty walkable space
- **YAML-based configuration** for terrain distribution ratios
- **Properties per tile**: walkable, transparent, edible

### AI Agents
- **Rodent AI** 🐀:
  - Searches surrounding tiles for food
  - Moves toward edible seedlings
  - Random movement when no food nearby
  - Hunger system
  - Collision detection with obstacles

### Development Tools
- **Type View Mode** (press `T`): Debug visualization showing:
  - `F` = Food (edible)
  - `O` = Obstacles (not walkable)
  - `R` = Rodent
  - `.` = Empty space
- **Pausable simulation** (press `SPACE`)
- **Performance metrics**: Load time displayed in window title

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Unpause simulation |
| `T` | Toggle type view (debug mode) |
| `Q` or `ESC` | Quit |

## Building

### Dependencies
- C++17 compiler (g++)
- ncurses library
- yaml-cpp library

**Install dependencies (Arch Linux):**
```bash
sudo pacman -S ncurses yaml-cpp
```

**Install dependencies (Ubuntu/Debian):**
```bash
sudo apt install libncurses-dev libyaml-cpp-dev
```

### Compile
```bash
make
```

### Run
```bash
./cursed_world
```

### Clean
```bash
make clean
```

## Configuration

Edit `config.yaml` to adjust terrain distribution:

```yaml
terrain:
  ratios:
    empty: 98      # Percentage of empty space
    seedlings: 2   # Percentage of seedlings (food)
    dead_trees: 3  # Percentage of dead trees
    rocks: 1       # Percentage of rocks
```

## Architecture

### Core Classes

- **Tile**: Terrain data with properties (walkable, edible, transparent)
- **TerminalMatrix**: 2D grid of tiles with rendering capabilities
- **Actuator**: Base class for dynamic entities
- **Rodent**: AI agent that navigates and eats food

### Design Pattern
- Tiles contain optional Actuator pointers for dynamic entities
- Matrix provides spatial queries for AI pathfinding
- YAML configuration separates data from code

## Project Structure

```
cursed_world/
├── main.cpp              # Entry point and simulation loop
├── TerminalMatrix.h/cpp  # Grid management and rendering
├── Tile.h/cpp           # Tile properties and behavior
├── Actuator.h/cpp       # Dynamic entity base class
├── Rodent.h/cpp         # Rodent AI implementation
├── config.h             # YAML configuration loader
├── config.yaml          # Terrain distribution settings
├── Makefile             # Build configuration
└── README.md
```

## Future Features

- [ ] More AI agent types (NPCs, traps, enemies)
- [ ] Player character with keyboard control
- [ ] Seedling growth/regrowth system
- [ ] Inventory and item system
- [ ] Multiple terrain layers (soil types)
- [ ] Save/load world state

## License

MIT License - See LICENSE file for details

## Development

Built with:
- C++17
- ncurses (terminal UI)
- yaml-cpp (configuration)
- UTF-8 emoji support

**Performance**: ~60 FPS simulation with real-time rendering
