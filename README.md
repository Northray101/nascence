# nascence

A neural 2D bacterial simulator. **Evolve** the *brains* of soft-bodied bacteria
— a whole population competes live, the best survive and breed, the rest are
culled — then give your species a name and spawn trained individuals into a
living top-down sandbox to watch them hunt for food by smell.

This is an early rough-draft build with all the major pieces sketched in.

## What you can do

1. **Create a species** and name it, choosing its **role** (forager that eats
   food, or predator that hunts prey), **body** (firm or squishy *jelly*),
   number of **legs**, and **leg joints** (simple, or hip + knee). More legs
   make a **longer, more elongated body** (centipede-like).
2. **Train its brain in a live sandbox** — a whole **population** of variations
   lives in the world at once. They all start from the **same spot each
   generation** and **don't bump into each other**; what changes is the
   **random environment**, which gets **harder every level** (goals farther
   away, more obstacles). Each generation the **best survive** and the rest are
   **culled and replaced by mutated copies of the winners**. Creatures sense
   their goal through an **invisible cone of vision** with limited range (plus
   smell), so they must *face* and get close to what they're after. Use
   **Advance level** to crank up difficulty yourself, or leave **Auto** on to
   let it level up once the species masters a level. Meanwhile you can
   *interfere*: drop/clear food, drag the current leader, draw walls, and give a
   **Treat (+)** / **Scold (–)**. A **speed slider** (up to 5,000×) and
   **pause** let you watch closely or fast-forward. (This is *neuroevolution* —
   fast to start and easy to watch, with no slow warm-up.)
3. **Save** the best brain automatically when you press *Save best & stop*.
4. **Spawn** trained creatures into the Sandbox, drop food, and watch foragers
   eat and predators hunt.

> Rough-draft note: foragers, jelly bodies, two-jointed legs, predators and the
> live influence tools are all in, but they need tuning and real play-testing —
> some behaviours will look rough until trained for a while.

## Setup (macOS) — one time

You need **Python 3.10, 3.11, or 3.12**. Very new versions (3.13 / 3.14) don't
have ready-to-install packages yet and will fail, so if you only have one of
those, install **Python 3.12** from
<https://www.python.org/downloads/macos/> (download the latest "Python 3.12.x"
installer). `setup.sh` checks this for you and tells you what to do.

Then:

1. Open the **Terminal** app.
2. Type `cd ` (with a space), then drag this project folder onto the Terminal
   window and press **Return**.
3. Run:

   ```bash
   bash setup.sh
   ```

   This builds a private `venv` folder with everything the app needs. It can take
   a few minutes the first time.

## Run it — the easy way

In **Finder**, double-click **`nascence.command`**. The first time, it sets
everything up automatically and then launches; after that it just launches.

> First time only: macOS may say it "cannot verify the developer". If so,
> **right-click `nascence.command` → Open → Open**. You only do this once.

(Prefer Terminal? `bash run.sh` does the same thing — and runs setup for you on
the first launch.)

A window opens. From the main menu:

- **Species** → *New species* → name it → *Create*. Select it, then *Train selected*.
- The training screen immediately spawns a **population** that starts wriggling
  and competing. Watch the **best-score line** climb generation by generation;
  drop food, drag the leader, or use **Treat / Scold** to steer it. Use the
  **speed slider** to fast-forward. When you like how it moves, press **Save
  best & stop** — the winning brain is saved.
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
