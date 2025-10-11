# Cat Evolution Implementation Session

**Date:** 2025-10-06
**Task:** Implement neural network-based evolution for cats in Cursed World

## Overview

Successfully implemented a complete evolutionary system for cats, allowing them to co-evolve with mice through neural network-based decision making and genetic algorithms.

## Architecture Changes

### Cat Neural Network
- **Architecture:** 10 inputs → 16 hidden (RNN) → 9 outputs
- **Inputs (10 total):**
  1-8. Surrounding tiles (NW, N, NE, W, E, SW, S, SE) encoded as:
    - 0.0 = Empty space
    - 0.125 = Plants
    - 0.25 = Seedlings
    - 0.375 = Dead trees
    - 0.5 = Rocks
    - 0.625 = Seeds
    - 0.75 = Rodent (prey!)
    - 1.0 = Other cat (avoid)
  9. Rodents eaten (normalized 0-1, max ~20)
  10. Eat cooldown (normalized 0-1, max 30 ticks)

- **Outputs (9 total):** Direction selection via argmax
  - 8 directional movements (NW, N, NE, W, E, SW, S, SE)
  - Stay in place

### Fitness Function
```cpp
fitness = (rodentsEaten × 100) + (age × 0.1)
```
Heavily weighted toward hunting success with minor survival bonus.

## Files Modified

### 1. include/Cat.h
**Added:**
- `NeuralNetwork brain` - 10→16→9 RNN
- `bool alive` - Alive/dead state
- `int id` - Unique identifier
- `static int nextId` - ID counter

**New Methods:**
- `Cat(int, int, string, vector<double>)` - Constructor with optional weights
- `getSurroundingInfo(TerminalMatrix&)` - Encode environment for NN
- `reproduce(int, int)` - Create mutated offspring
- `kill()` - Mark cat as dead
- `resetMemory()` - Reset RNN hidden state
- `getBrain()` - Access neural network
- `getId()`, `isAlive()` - Getters

### 2. src/Cat.cpp
**Modified Constructor:**
- Initialize brain with 10→16→9 architecture
- Accept optional weight vector
- Random initialization if no weights provided

**New Method: `getSurroundingInfo()`**
- Scans 8 surrounding tiles
- Encodes terrain types and actuators
- Normalizes all inputs to 0-1 range
- Returns 10-element vector

**Updated `update()` Method:**
- Replaced hardcoded AI with neural network
- Gets environment info via `getSurroundingInfo()`
- Forward pass through brain
- Argmax to select best action
- Maps action to movement direction

**New Method: `reproduce()`**
- Clones parent's brain weights
- Applies 5% mutation rate
- Mutation amount: ±0.3
- Returns new Cat with mutated brain

### 3. include/PopulationManager.h
**Updated:**
- `initializeCats(int, TerminalMatrix&, vector<double>)` - Added weights parameter

**Added:**
- `evolveCats(TerminalMatrix&)` - Cat evolution logic
- `getBestCat()` - Returns highest fitness cat

### 4. src/PopulationManager.cpp
**Updated `initializeCats()`:**
- Accept optional best weights
- Pass weights to Cat constructor
- Apply 20% mutation for initial diversity

**New Method: `evolveCats()`:**
- Sort cats by fitness (descending)
- Keep top 50% as parents (more selective than mice)
- Clear old generation from tiles
- Respawn parents with new positions
- Reset rodentsEaten and memory
- Create offspring with 15% mutation (±0.4)
- Target: maintain initial cat count

**Updated `evolveGeneration()`:**
- Call `evolveCats()` at start of generation cycle
- Ensures cats evolve alongside mice

**Removed:**
- Mid-generation cat reproduction (lines 130-176)
- Cats now only evolve between generations

**New Method: `getBestCat()`:**
- Iterates through cats
- Returns cat with highest fitness
- Returns nullptr if no cats exist

### 5. src/main.cpp
**Brain Loading:**
- Added `bestCatWeights` vector
- Load from `best_cat_brain.dat` on startup
- 10→16→9 architecture check
- Error handling with fallback to random

**Initialization:**
- Pass `bestCatWeights` to `initializeCats()`
- Cats start with evolved brains from previous session

**Brain Saving:**
- Get best cat via `getBestCat()`
- Save to `best_cat_brain.dat` on exit
- Confirmation message on success

## Evolution Parameters

### Cat Evolution
- **Selection Rate:** Top 50% (vs 20% for mice)
- **Mutation Rate:** 15% of weights (vs 10% for mice)
- **Mutation Amount:** ±0.4 (vs ±0.5 for mice)
- **Reproduction Mutation:** 5% / ±0.3 (vs 5% / ±0.3 for mice)

### Rationale
- **Higher selection pressure:** Cats must be effective hunters
- **Moderate mutation:** Balance exploration vs exploitation
- **Generation-based only:** Prevents exponential population growth
- **Fitness reset:** Each generation starts fresh for fair evaluation

## Co-Evolution Dynamics

### Predator-Prey Arms Race
1. **Mice evolve** better food-finding and cat-avoidance strategies
2. **Cats evolve** better hunting and mouse-tracking strategies
3. **Feedback loop:** As mice get better, cats must adapt; vice versa
4. **Emergent behavior:** Complex hunting patterns should emerge

### Expected Outcomes
- Early generations: Random movement
- Mid generations: Basic chase behavior
- Late generations: Anticipatory movement, cutting off escape routes
- Stable state: Dynamic equilibrium between predator/prey effectiveness

## File Artifacts

### New Files Created
- `best_cat_brain.dat` - Saved neural network weights (binary)

### Modified Files
- `include/Cat.h` - Added NN and evolution support
- `src/Cat.cpp` - Implemented NN decision-making
- `include/PopulationManager.h` - Added cat evolution methods
- `src/PopulationManager.cpp` - Implemented cat evolution logic
- `src/main.cpp` - Added brain persistence

## Build Status
✅ **Compilation successful** (with warnings about signed/unsigned comparison)
- All core functionality implemented
- No blocking errors
- Ready for testing

## Testing Recommendations

### Short-term (1-5 generations)
- Verify cats move using NN decisions
- Check fitness calculation
- Confirm evolution between generations
- Validate brain save/load

### Medium-term (10-20 generations)
- Monitor fitness trends (should increase)
- Observe hunting behavior improvements
- Check for population stability
- Verify mutation diversity

### Long-term (50+ generations)
- Evaluate emergent hunting strategies
- Compare cat fitness across sessions
- Analyze co-evolution with mice
- Test brain transfer between runs

## Future Enhancements

### Potential Improvements
- [ ] Separate cat death logging (like mice)
- [ ] Cat-specific stats in dashboard (avg kills, best hunter)
- [ ] Visualization of cat decision-making
- [ ] Configurable evolution parameters via YAML
- [ ] Multiple cat species with different strategies
- [ ] Energy system for cats (tiredness after hunting)
- [ ] Pack hunting behavior (multi-cat cooperation)

### Research Opportunities
- Compare evolution speed: cats vs mice
- Test impact of different mutation rates
- Analyze learned hunting patterns
- Study equilibrium population dynamics

## Key Insights

### Design Decisions
1. **10 inputs vs 9:** Cats need eat cooldown awareness to avoid failed attacks
2. **50% selection:** More aggressive than mice to maintain hunting effectiveness
3. **No mid-gen reproduction:** Ensures fair fitness evaluation
4. **Separate brain file:** Allows independent evolution tracking

### Implementation Challenges
1. **Terrain enum naming:** Fixed `SEEDS` → `SEED`
2. **Signed/unsigned warnings:** Non-critical, can be ignored
3. **Population management:** Careful tile cleanup during evolution
4. **Memory resets:** Important for fair generation comparison

## Usage

### Running the Simulation
```bash
make clean && make
./evolution
```

### Controls
- `SPACE` - Pause/unpause
- `T` - Toggle type view
- `Q/ESC` - Quit and save brains
- Arrow keys - Move ghost

### Monitoring Evolution
- Watch dashboard for cat count and mouse population
- Check `best_cat_brain.dat` file size (should be consistent)
- Observe hunting behavior changes over generations
- Compare fitness values between sessions

## Conclusion

Cat evolution is now fully implemented and integrated with the existing mouse evolution system. The simulation creates a true co-evolutionary environment where both predators and prey adapt to each other's strategies, leading to increasingly sophisticated behaviors over time.

**Status:** ✅ Implementation Complete
**Next Steps:** Run multi-generation tests, collect behavioral data, document emergent strategies
