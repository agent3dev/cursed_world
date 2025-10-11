# Cursed World 🌍

A collection of terminal-based simulations and games built with C++17 and ncurses.

## Project Structure

```
cursed_world/
├── common/              # Shared components across all games
│   ├── include/        # Common headers (TerminalMatrix, Menu, Tile, etc.)
│   └── src/            # Common source files
├── games/               # Individual games/simulations
│   ├── evolution/      # Evolution simulation (Mice vs Cats with NNs)
│   └── city_escape/    # City escape game (coming soon)
├── Makefile            # Main build system
├── cursed_world        # Main executable (generated)
└── README.md
```

## Games

### 1. Evolution Simulation 🧬
Location: `games/evolution/`

An evolutionary simulation where AI-controlled mice learn to survive, find food, and avoid cat predators through neural network evolution.

**Features:**
- Recurrent Neural Networks (9→16→9 for mice, 10→16→9 for cats)
- Co-evolution between predator and prey
- Genetic algorithms with mutation
- Persistent brain saving/loading
- Real-time statistics

**Controls:**
- `SPACE` - Pause/Unpause
- `T` - Toggle type view
- `Arrow Keys` - Move ghost character
- `Q` or `ESC` - Quit and save

[See Evolution README](./games/evolution/README.md) for details.

### 2. City Scape 🏙️
Location: `games/city_scape/`

An urban navigation simulation where autonomous citizens and vehicles navigate through a procedurally generated city.

**Features:**
- Procedurally generated cities with roads, buildings, parks
- Autonomous citizens with pathfinding AI
- Vehicle traffic simulation
- Real-time urban dynamics

**Controls:**
- `SPACE` - Pause/Unpause
- `T` - Toggle type view
- `Q` or `ESC` - Quit

[See City Scape README](./games/city_scape/README.md) for details.

## Building

### Build Everything
```bash
make
```

### Run
```bash
./cursed_world
```

This will present a menu where you can select which game to play.

### Clean
```bash
make clean
```

## Dependencies

- **C++17 compiler** (g++)
- **ncurses** (terminal UI)
- **yaml-cpp** (configuration)

**Install on Arch Linux:**
```bash
sudo pacman -S ncurses yaml-cpp base-devel
```

**Install on Ubuntu/Debian:**
```bash
sudo apt install build-essential libncurses-dev libyaml-cpp-dev
```

## Adding a New Game

1. Create a new directory in `games/your_game/`
2. Add `include/` and `src/` subdirectories
3. Include common headers from `../../common/include/`
4. Add your game to the main Makefile
5. Register it in the main menu

## Common Components

The `common/` directory contains reusable components:

- **TerminalMatrix**: Grid-based ncurses rendering
- **Tile**: Individual grid cells with properties
- **Menu**: ncurses menu system
- **Actuator**: Base class for dynamic entities
- **Simulation**: Base class for game simulations

## License

MIT License

---

**Happy gaming!** 🎮
