# Cursed World Evolution 🌍🧬

A terminal-based evolutionary simulation where AI-controlled mice learn to survive, find food, and avoid predators through neural network evolution.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

## Overview

Cursed World is an evolutionary simulation featuring:
- **Recurrent Neural Network Mice**: Each mouse has a brain (9→16→9 RNN) with memory that evolves through generations
- **Evolving Cat Predators**: Cats with neural networks (10→16→9) that co-evolve alongside mice
- **Genetic Algorithm**: The fittest mice and cats survive and pass on their genes with mutations
- **Predator-Prey Dynamics**: Cats hunt mice at equal speed, creating survival pressure
- **Real-time Evolution**: Watch both species learn optimal strategies over generations
- **Persistent Learning**: Save/load the best brains for both species between sessions
- **Ecosystem Dynamics**: Plant regrowth system with seed dispersal
- **Debug Logging**: Track every death with detailed statistics

## Features

### 🧠 Recurrent Neural Network Architecture

#### Mouse Brain (9→16→9 RNN)

**Inputs (9):**
- 8 surrounding tiles encoded as terrain type:
  - `0` = Empty space
  - `1` = Plants
  - `2` = Seedlings (food)
  - `3` = Dead trees (obstacle)
  - `4` = Rocks (obstacle)
  - `5` = Seeds (will grow into seedlings)
  - `6` = Cat (danger!)
  - `7` = Wall/out of bounds
- Current energy level (normalized 0-1)

**Hidden Layer:** 16 neurons with recurrent connections (memory)
- 16×16 recurrent weight matrix provides short-term memory
- Hidden state persists across timesteps
- Enables temporal pattern learning and context awareness

**Outputs (9):**
- 8 directional movements (NW, N, NE, W, E, SW, S, SE)
- Stay in place
- Uses argmax to select the highest-valued action

#### Cat Brain (10→16→9 RNN)

**Inputs (10):**
- 8 surrounding tiles encoded as terrain type:
  - `0` = Empty space
  - `1` = Plants
  - `2` = Seedlings
  - `3` = Dead trees
  - `4` = Rocks
  - `5` = Seeds
  - `6` = Rodent (prey!)
  - `7` = Wall/out of bounds
  - `8` = Other cats (avoid)
- Rodents eaten count (normalized to 0-1)
- Eat cooldown status (normalized 0-1)

**Hidden Layer:** 16 neurons with recurrent connections

**Outputs (9):**
- Same as mouse: 8 directions + stay
- Learns hunting patterns and patrol strategies

### 🌱 Terrain & Ecosystem System

- **Tile-based world** with procedurally generated terrain
- **YAML configuration** for terrain distribution ratios
- **Dynamic food system**: Mice must find and eat seedlings 🌱 to survive
- **Plant regrowth**: Mice poop seeds 🔸 every 50 ticks, which grow into seedlings after 100 ticks
- **Obstacles**: Dead trees 🪾, rocks 🪨, and tombstones 🪦 block movement
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

### 🐈 Cat AI Behavior

Cats learn to:
- **Hunt efficiently** by tracking and chasing mice
- **Navigate around obstacles** and terrain
- **Coordinate movements** to avoid other cats
- **Optimize kill timing** with eat cooldown management
- **Develop patrol strategies** when no prey is nearby

**Mechanics:**
- **Neural Network Controlled**: Cats use their brain to decide movement
- **Vision Range**: Can see 3 tiles in all directions (7x7 area)
- **Kill Instantly**: Entering a mouse tile kills it
- **Eat Cooldown**: 30 ticks between kills
- **Move at same speed** as mice (creating strong selection pressure)
- **Never die**: Only evolve between generations

**Fitness Function:**
```
fitness = (rodents_eaten × 100) + (age × 0.1)
```

**Evolution:**
- Top 50% of cats selected as parents each generation
- Offspring created with 15% mutation rate (±0.4 adjustment)
- Population maintained at 3 cats (configurable)
- Memory reset between generations

### 🧬 Evolution System

**Generation Cycle:**
1. Run for 2000 ticks or until all mice die (no respawning)
2. Sort both mice and cats by fitness
3. **Mice**: Keep top 20% as parents
4. **Cats**: Keep top 50% as parents
5. Create offspring with mutation for both species
6. Reset to 80 mice and 3 cats for next generation
7. Clear all tombstones and reset positions

**Brain Persistence:**
- Best mouse brain saved to `best_brain.dat` on exit
- Best cat brain saved to `best_cat_brain.dat` on exit
- Both loaded on startup for continuous evolution across sessions
- Includes recurrent weights for memory

**Mutation Rates:**
- **Mice** live reproduction: 5% rate, ±0.3 adjustment
- **Mice** generation evolution: 10% rate, ±0.5 adjustment
- **Cats** generation evolution: 15% rate, ±0.4 adjustment

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

## Debug Logging

The simulation automatically creates `death_log.txt` with detailed death statistics:

**CSV Format:**
```
Generation,Tick,MouseID,PosX,PosY,Energy,Age,FoodEaten,Cause
1,247,42,15,8,0.00,247,3,Starvation
1,312,28,22,14,45.30,312,5,Eaten
```

**Fields:**
- **Generation/Tick**: When the death occurred
- **MouseID**: Unique identifier for each mouse
- **PosX/PosY**: Death location coordinates
- **Energy**: Energy level at death
- **Age**: Survival time in ticks
- **FoodEaten**: Number of seedlings consumed
- **Cause**: "Starvation" (energy ≤ 0) or "Eaten" (killed by cat)

**Notes:**
- Log appends across sessions with timestamps
- Tombstones (🪦) only placed for starvation deaths
- File automatically created on first run

## Project Structure

```
cursed_world/
├── main.cpp              # Entry point and simulation loop
├── TerminalMatrix.h/cpp  # Grid management and ncurses rendering
├── Tile.h/cpp            # Tile properties and terrain types
├── Actuator.h/cpp        # Base class for dynamic entities
├── Rodent.h/cpp          # Mouse AI with neural network
├── Cat.h/cpp             # Cat predator AI with neural network
├── Ghost.h/cpp           # Player-controlled ghost
├── NeuralNetwork.h/cpp   # Recurrent neural network with memory
├── PopulationManager.h/cpp # Evolution and population management
├── config.h              # YAML configuration loader
├── config.yaml           # Terrain distribution settings
├── Makefile              # Build configuration
├── best_brain.dat        # Saved mouse neural network (generated)
├── best_cat_brain.dat    # Saved cat neural network (generated)
├── death_log.txt         # Death statistics CSV (generated)
└── README.md
```

## How It Works

### Neural Networks

**Mice** have a recurrent neural network:
- **Layer 1**: 9 inputs → 16 neurons (tanh activation)
- **Recurrent**: 16×16 hidden state matrix (memory)
- **Layer 2**: 16 neurons → 9 outputs (tanh activation)
- **Decision**: Argmax selects the highest output

**Cats** have a recurrent neural network:
- **Layer 1**: 10 inputs → 16 neurons (tanh activation)
- **Recurrent**: 16×16 hidden state matrix (memory)
- **Layer 2**: 16 neurons → 9 outputs (tanh activation)
- **Decision**: Argmax selects the highest output

### Co-Evolution

1. **Initialization**: 80 mice and 3 cats spawn with random or loaded brain weights
2. **Simulation**: Both species interact for 2000 ticks
   - Mice eat food, reproduce, avoid cats, die from starvation or being eaten
   - Cats hunt mice, accumulate kills, never die mid-generation
3. **Selection**:
   - Top 20% of mice survive based on fitness
   - Top 50% of cats survive based on fitness
4. **Reproduction**: Both species create mutated offspring
   - Mice reach 80 population
   - Cats reach 3 population
5. **Repeat**: New generation begins with both species evolved

### Mutation

**Mice:**
- **Live reproduction**: 5% rate, ±0.3 adjustment
- **Generation evolution**: 10% rate, ±0.5 adjustment

**Cats:**
- **Generation evolution**: 15% rate, ±0.4 adjustment
- Higher mutation rate encourages hunting diversity

## Performance

- **Target**: ~60 FPS with 80 mice and 3 cats
- **Evolution Speed**: Typically sees improvement within 5-10 generations for both species
- **Memory**: Efficient smart pointer management prevents leaks

## Tips for Training

1. **Start fresh**: Delete `best_brain.dat` and `best_cat_brain.dat` to reset evolution
2. **Let it run**: Co-evolution takes time—at least 20+ generations for interesting behaviors
3. **Watch patterns**:
   - Successful mice cluster near food sources and flee from cats
   - Successful cats develop patrol patterns and chase strategies
4. **Cat pressure**: More cats = stronger selection for mouse evasion
5. **Energy awareness**: Mice should learn to prioritize food when low energy
6. **Arms race**: As mice evolve better evasion, cats evolve better hunting

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

- [x] Seedling regrowth system (seed dispersal via poop)
- [x] Recurrent neural networks (memory via hidden state)
- [x] Death tracking and debugging tools
- [ ] Multiple terrain layers (temperature, moisture)
- [ ] Mouse communication/cooperation
- [ ] Visualization of neural network decisions
- [ ] Save/load world state
- [ ] Configurable evolution parameters
- [ ] Real-time statistics dashboard

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
