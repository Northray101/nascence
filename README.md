# nascence

A neural 2D bacterial simulator. Train the *brains* of soft-bodied bacteria with
reinforcement learning, give your species a name, then spawn trained individuals
into a living top-down sandbox and watch them hunt for food by smell.

This is the **Phase 1** vertical slice: create a species → train it to crawl
toward food → save it → spawn it in the sandbox.

## What you can do today

1. **Create a species** and name it (e.g. "Wiggler"), choosing how many legs it has.
2. **Train its brain** with one click — watch a chart of its reward climb as it
   learns to move its legs and follow the smell of food. Training uses PPO
   (a standard reinforcement-learning algorithm) under the hood.
3. **Save** the trained brain automatically.
4. **Spawn** trained bacteria into the sandbox, **drop food** with your mouse, and
   watch them crawl over and eat it.

Coming in later phases: jiggly soft "jelly" bodies, trainable enemies, smarter
two-jointed legs, and more.

## Setup (macOS) — one time

You need **Python 3** (get it from <https://www.python.org/downloads/> if you
don't have it). Then:

1. Open the **Terminal** app.
2. Type `cd ` (with a space), then drag this project folder onto the Terminal
   window and press **Return**.
3. Run:

   ```bash
   bash setup.sh
   ```

   This builds a private `venv` folder with everything the app needs. It can take
   a few minutes the first time.

## Run it

```bash
bash run.sh
```

A window opens. From the main menu:

- **Species** → *New species* → name it → *Create*. Select it, then *Train selected*.
- On the training screen pick **Quick / Normal / Thorough** and press **Start**.
  (The very first start takes a few seconds while it loads PyTorch.) Watch the
  reward line rise, then it saves automatically.
- Go **Back**, then from the main menu open the **Sandbox**. Pick your species,
  press **Spawn mode**, and click in the world to place one. Switch to **Food
  mode** and click to drop food. Watch it forage.

Saved species live in `~/Library/Application Support/nascence/species/`.

## For developers

```
nascence/            # the Python package
  sim/               # physics + world (no RL, no rendering)
  rl/                # Gymnasium env, PPO trainer, save/load
  species/           # on-disk species records (metadata + policy.zip)
  render/            # pygame drawing
  ui/                # pygame_gui screens + app loop
tests/               # sim + env contract tests
```

Run the fast tests (need only `numpy gymnasium pymunk pytest`):

```bash
python -m pytest tests/ -q
```

See `nascence/config.py` for all tunable constants.
