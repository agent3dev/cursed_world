# Cursed World Evolution 🌍🧬

A terminal-based evolutionary simulation where AI-controlled mice learn to survive, find food, and avoid predators through neural network evolution.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

## Overview

Cursed World is an evolutionary simulation featuring:
- **Neural Network Mice**: Each mouse has a brain (9→16→9 architecture) that evolves through generations
- **Genetic Algorithm**: The fittest mice survive and pass on their genes with mutations
- **Predator-Prey Dynamics**: Cats hunt mice, creating survival pressure
- **Real-time Evolution**: Watch mice learn optimal strategies over generations
- **Persistent Learning**: Save/load the best brain between sessions

## Features

### 🧠 Neural Network Architecture

**Inputs (9):**
- 8 surrounding tiles encoded as terrain type:
  - `0` = Empty space
  - `1` = Plants
  - `2` = Seedlings (food)
  - `3` = Dead trees (obstacle)
  - `4` = Rocks (obstacle)
  - `5` = Cat (danger!)
  - `6` = Wall/out of bounds
- Current energy level (normalized 0-1)

**Hidden Layer:** 16 neurons

**Outputs (9):**
- 8 directional movements (NW, N, NE, W, E, SW, S, SE)
- Stay in place
- Uses argmax to select the highest-valued action

### 🌱 Terrain System

- **Tile-based world** with procedurally generated terrain
- **YAML configuration** for terrain distribution ratios
- **Dynamic food system**: Mice must find and eat seedlings 🌱 to survive
- **Obstacles**: Dead trees 🪾 and rocks 🪨 block movement
- **Animated borders**: Alternating ⬛⬜ pattern

### 🐀 Mouse AI Behavior

Mice learn to:
- **Find food** when energy is low
- **Avoid obstacles** and walls
- **Flee from cats** when detected nearby
- **Conserve energy** by staying still when appropriate
- **Reproduce** when energy is high (≥120)

**Energy System:**
- Start with 100 energy
- Lose 0.2 energy per movement
- Lose 0.05 energy per tick (passive drain)
- Gain 40 energy from eating food
- Die at 0 energy (starvation)

**Fitness Function:**
```
fitness = (food_eaten × 10) + (age × 0.1)
```

### 🐈 Cat Behavior

- **Chase nearest mouse** within 7-tile radius
- **Enter mouse tiles** to kill them instantly
- **Patrol randomly** when no mice nearby
- **Reproduce** after eating 10 mice (max 20 cats)
- **Move slower** than mice (3-tick cooldown)

### 🧬 Evolution System

**Generation Cycle:**
1. Run for 2000 ticks or until all mice die
2. Sort mice by fitness
3. Keep top 20% as parents
4. Create offspring with 10% mutation rate (±0.5 weight adjustment)
5. Reset to 80 mice for next generation

**Brain Persistence:**
- Best brain automatically saved to `best_brain.dat` on exit
- Loaded on startup for continuous evolution across sessions

### 👻 Player Ghost

- Control a ghost with arrow keys
- Kill nearby mice by moving adjacent to them
- Useful for testing mouse avoidance behavior

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Unpause simulation |
| `T` | Toggle type view (debug mode showing F/O/R/C) |
| `Arrow Keys` | Move ghost character |
| `Q` or `ESC` | Quit and save best brain |

## Building

### Dependencies

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

### Compile

```bash
make
```

### Run

```bash
./cursed_world
```

The simulation starts paused. Press `SPACE` to begin.

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
    dead_trees: 3  # Percentage of dead trees (obstacles)
    rocks: 1       # Percentage of rocks (obstacles)
```

## Project Structure

```
cursed_world/
├── main.cpp              # Entry point and simulation loop
├── TerminalMatrix.h/cpp  # Grid management and ncurses rendering
├── Tile.h/cpp            # Tile properties and terrain types
├── Actuator.h/cpp        # Base class for dynamic entities
├── Rodent.h/cpp          # Mouse AI with neural network
├── Cat.h/cpp             # Cat predator AI
├── Ghost.h/cpp           # Player-controlled ghost
├── NeuralNetwork.h/cpp   # Feed-forward neural network
├── PopulationManager.h/cpp # Evolution and population management
├── config.h              # YAML configuration loader
├── config.yaml           # Terrain distribution settings
├── Makefile              # Build configuration
├── best_brain.dat        # Saved neural network weights (generated)
└── README.md
```

## How It Works

### Neural Network

Each mouse has a fully-connected feed-forward neural network:
- **Layer 1**: 9 inputs → 16 neurons (tanh activation)
- **Layer 2**: 16 neurons → 9 outputs (tanh activation)
- **Decision**: Argmax selects the highest output

### Evolution

1. **Initialization**: 100 mice spawn with random brain weights
2. **Simulation**: Mice move, eat, reproduce, and die for 2000 ticks
3. **Selection**: Top 20% survive based on fitness (food eaten + survival time)
4. **Reproduction**: Survivors create mutated offspring to reach 80 mice
5. **Repeat**: New generation begins

### Mutation

- **Rate**: 10% of weights are mutated each generation
- **Amount**: Random adjustment of ±0.5 per mutated weight
- **Diversity**: Prevents local optima and explores new strategies

## Performance

- **Target**: ~60 FPS with 100 mice and 5-20 cats
- **Evolution Speed**: Typically sees improvement within 5-10 generations
- **Memory**: Efficient smart pointer management prevents leaks

## Tips for Training

1. **Start fresh**: Delete `best_brain.dat` to reset evolution
2. **Let it run**: Evolution takes time—at least 10+ generations
3. **Watch patterns**: Successful mice cluster near food sources
4. **Cat pressure**: More cats = stronger selection for evasion
5. **Energy awareness**: Mice should learn to prioritize food when low energy

## Troubleshooting

**Terminal too small?**
- The simulation adapts to terminal size (divides width by 2 for emojis)
- Minimum recommended: 80×24 characters

**Mice not learning?**
- Check if mutation rate is too high/low in `PopulationManager.cpp:397`
- Verify fitness function rewards desired behavior
- Ensure enough generations have passed

**Performance issues?**
- Reduce population size in `main.cpp:136` (currently 100)
- Increase tick delay in `main.cpp:227` (currently 100ms)

## Future Ideas

- [ ] Seedling regrowth system
- [ ] Multiple terrain layers (temperature, moisture)
- [ ] Mouse communication/cooperation
- [ ] Recurrent neural networks (memory)
- [ ] Visualization of neural network decisions
- [ ] Save/load world state
- [ ] Configurable evolution parameters

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- C++17
- ncurses (terminal UI)
- yaml-cpp (configuration)
- UTF-8 emoji support

---

**Happy evolving!** 🧬🐀🐈
