# City Scape 🏙️

A terminal-based urban navigation simulation where citizens and vehicles navigate through a procedurally generated city.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)

## Overview

City Scape is an urban navigation simulation featuring:
- **Procedurally Generated Cities**: Each run creates a unique city layout with roads, buildings, parks, and more
- **Autonomous Citizens**: Pedestrians navigate from point A to point B using simple pathfinding
- **Vehicle Traffic**: Cars and vehicles move through the city following roads
- **Real-time Simulation**: Watch the city come alive with movement and activity
- **Terminal-based Rendering**: Beautiful emoji-based graphics in your terminal

## Features

### 🏗️ City Terrain System

The city is composed of different terrain types:

- **🏢 Buildings** (not walkable) - Office buildings, shops, hotels, banks
- **⬛ Roads** - Streets for vehicle traffic
- **⬜ Sidewalks** - Walkways for pedestrians
- **🌳 Parks** - Green spaces with trees
- **💧 Water** - Fountains and water features
- **🅿️ Parking** - Parking lots
- **🚸 Crosswalks** - Pedestrian crossings
- **🚦 Intersections** - Traffic light locations

### 🚶 Citizens (Pedestrians)

Citizens are AI-controlled pedestrians that:
- **Navigate to destinations** using simple pathfinding
- **Avoid obstacles** like buildings and water
- **Stay on walkable terrain** (sidewalks, roads)
- **Give up if stuck** with a patience system
- **Reach their goals** and then select new destinations

**Pathfinding:**
- Simple greedy algorithm moving towards destination
- Avoids non-walkable tiles
- Tries alternate routes when blocked
- Patience decreases when stuck

### 🚗 Vehicles

Vehicles navigate the city streets:
- **Follow directions** (North, South, East, West)
- **Stop at obstacles** and intersections
- **Turn at corners** when blocked
- **Avoid collisions** with other entities
- **Maintain traffic flow** with stop counters

**Movement:**
- Directional movement along roads
- Intersection handling with turns
- Stop for 3 ticks at intersections
- Avoid other vehicles and pedestrians

### ⚙️ Configuration

City generation is controlled by ratios in `CityTerrain.h`:

```cpp
config.road = 40;       // 40% roads
config.sidewalk = 30;   // 30% sidewalks
config.building = 20;   // 20% buildings
config.park = 5;        // 5% parks
config.water = 3;       // 3% water features
config.parking = 2;     // 2% parking lots
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Unpause simulation |
| `T` | Toggle type view (debug mode) |
| `Q` or `ESC` | Quit simulation |

## Building

### Dependencies

- **C++17 compiler** (g++)
- **ncurses** (terminal UI)

**Install on Arch Linux:**
```bash
sudo pacman -S ncurses base-devel
```

**Install on Ubuntu/Debian:**
```bash
sudo apt install build-essential libncurses-dev
```

### Compile

```bash
cd games/city_scape
make
```

### Run

```bash
./city_scape
```

Or from the main Cursed World menu:
```bash
cd ../..
./cursed_world
# Select "City Scape" from menu
```

### Clean

```bash
make clean
```

## Project Structure

```
city_scape/
├── include/
│   ├── CityScapeSimulation.h  # Main simulation class
│   ├── CityTerrain.h           # Terrain types and configuration
│   ├── Citizen.h               # Pedestrian AI
│   └── Vehicle.h               # Vehicle AI
├── src/
│   ├── CityScapeSimulation.cpp # Simulation implementation
│   ├── CityTerrain.cpp         # Terrain config
│   ├── Citizen.cpp             # Pedestrian logic
│   ├── Vehicle.cpp             # Vehicle logic
│   └── main.cpp                # Entry point
├── build/                      # Build artifacts (generated)
├── Makefile                    # Build configuration
└── README.md
```

## How It Works

### Simulation Loop

1. **Initialization**: Generate city terrain procedurally
2. **Spawn Entities**: Create initial citizens and vehicles
3. **Update Loop**: Each tick:
   - Update all citizens (pathfinding)
   - Update all vehicles (movement)
   - Render the city
   - Handle user input
4. **Cleanup**: Exit gracefully

### Citizen Pathfinding

Citizens use a simple greedy pathfinding algorithm:

1. Calculate delta to destination (dx, dy)
2. Move in the direction of larger delta
3. If blocked, try alternate route
4. Decrease patience counter
5. Give up if patience runs out

### Vehicle Navigation

Vehicles follow a directional movement system:

1. Move in current direction
2. Check if path is clear ahead
3. If blocked, turn right (intersection logic)
4. Stop for N ticks after turning
5. Continue moving

## Future Ideas

- [ ] Traffic lights with timing system
- [ ] Citizens waiting at crosswalks
- [ ] Different vehicle types (buses, bikes)
- [ ] Rush hour simulation with increased traffic
- [ ] Public transit system (buses, trains)
- [ ] Building interiors citizens can enter
- [ ] Weather effects (rain, snow)
- [ ] Day/night cycle
- [ ] Pathfinding improvements (A* algorithm)
- [ ] Traffic congestion metrics
- [ ] Save/load city layouts

## Performance

- **Target**: ~60 FPS with moderate city size
- **Entities**: 10 citizens + 5 vehicles by default
- **Scalable**: Adjust spawn counts for different performance
- **Update Rate**: 150ms per tick (slower than evolution)

## Tips

1. **Watch the flow**: Observe how citizens navigate around obstacles
2. **Traffic patterns**: Vehicles create interesting movement patterns
3. **Stuck entities**: Notice when citizens give up (patience system)
4. **Urban planning**: Buildings naturally create channels for movement
5. **Experiment**: Modify terrain ratios to create different city types

## Troubleshooting

**Terminal too small?**
- The simulation adapts to terminal size (divides width by 2 for emojis)
- Minimum recommended: 80×24 characters

**Entities not moving?**
- Check if they're stuck on non-walkable terrain
- Verify patience counter hasn't run out
- Ensure destinations are reachable

**Performance issues?**
- Reduce entity counts in `initializeTerrain()`
- Increase tick delay in `run()` (currently 150ms)

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- C++17
- ncurses (terminal UI)
- UTF-8 emoji support

Part of the **Cursed World** simulation framework.

---

**Happy navigating!** 🚶🚗🏙️
