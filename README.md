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

### 1. Evolution Simulation 🧬🚀
Location: `games/evolution/`

An evolutionary simulation where AI-controlled mice learn to survive, find food, and avoid cat predators through neural network evolution. **Features GPU acceleration via CUDA for massive performance gains!**

**Features:**
- **CUDA GPU Acceleration** - 10-20× speedup for large populations
- Recurrent Neural Networks (9→16→9 for mice, 10→16→9 for cats)
- Co-evolution between predator and prey
- **Overpopulation Mode** - Start with 45 mice + 5 cats for aggressive evolution
- Genetic algorithms with mutation
- Persistent brain saving/loading
- Real-time statistics and backend selection

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

### Build Everything (CPU Only)
```bash
make
```

### Build with CUDA GPU Acceleration 🚀
```bash
make ENABLE_CUDA=1
```

**Requirements for CUDA build:**
- NVIDIA GPU with compute capability 6.0+
- CUDA Toolkit 11.0+ installed
- Driver version 450+

### Run
```bash
./cursed_world
```

This will present a menu where you can select which game to play.

### Run Evolution Directly with GPU
```bash
cd games/evolution
./evolution --backend=cuda
```

**Backend Options:**
- `--backend=cpu` - Force CPU backend
- `--backend=cuda` - Force CUDA GPU backend
- `--backend=auto` - Auto-select based on population (default)
- `--auto-threshold=N` - Use GPU when population ≥ N agents

### Clean
```bash
make clean
```

## Dependencies

### Required
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

### Optional (for GPU acceleration)
- **CUDA Toolkit** 11.0 or later
- **NVIDIA GPU** with compute capability 6.0+

**Install CUDA on Arch Linux:**
```bash
sudo pacman -S cuda
```

**Install CUDA on Ubuntu:**
```bash
# See: https://developer.nvidia.com/cuda-downloads
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
- **ComputeBackend**: Abstract interface for CPU/GPU compute
  - **CPUBackend**: Standard CPU implementation (always available)
  - **CUDABackend**: GPU-accelerated implementation (optional)
- **NeuralNetwork**: Recurrent neural network with GPU support

## License

MIT License

---

**Happy gaming!** 🎮
