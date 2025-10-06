# Cursed World Simulations 🌍

A collection of terminal-based simulation games built with C++17 and ncurses.

## Simulations

### 🧬 [Evolution](./evolution/)
An evolutionary simulation where AI-controlled mice learn to survive, find food, and avoid predators through neural network evolution.

- **Neural Network Architecture**: 9→16(recurrent)→9 with RNN memory
- **Genetic Algorithm**: Fitness-based selection with mutation
- **Ecosystem**: Predator-prey dynamics with plant regrowth system
- **Real-time Evolution**: Watch mice learn optimal strategies over generations

[See Evolution README](./evolution/README.md) for details.

## Requirements

- **C++17 compiler** (g++)
- **ncurses** (terminal UI)
- **yaml-cpp** (configuration)

### Install Dependencies

**Arch Linux:**
```bash
sudo pacman -S ncurses yaml-cpp base-devel
```

**Ubuntu/Debian:**
```bash
sudo apt install build-essential libncurses-dev libyaml-cpp-dev
```

## Project Structure

```
cursed_world/
├── evolution/           # Evolution simulation
│   ├── src/            # Source files
│   ├── include/        # Header files
│   ├── build/          # Build artifacts (generated)
│   ├── config.yaml     # Configuration
│   ├── Makefile        # Build system
│   └── README.md       # Documentation
└── README.md           # This file
```

## Building

Navigate to a simulation folder and run:

```bash
cd evolution
make
./evolution
```

## License

MIT License - See individual simulation folders for details.

---

**More simulations coming soon!** 🎮
