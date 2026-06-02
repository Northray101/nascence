"""Global constants and tuning defaults.

Everything a non-technical user might want nudged later lives here as a named
value, so the GUI/sliders have a single source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Window / rendering
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
WINDOW_TITLE = "nascence"
FPS = 60

# World is measured in "micrometres" (arbitrary sim units). The camera maps
# these to screen pixels.
WORLD_WIDTH = 1200.0
WORLD_HEIGHT = 1200.0

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
# Fixed simulation timestep (seconds). We step the physics in small substeps
# for stability regardless of frame rate.
SIM_DT = 1.0 / 60.0
PHYSICS_SUBSTEPS = 3

# Top-down "fluid": Pymunk global damping = fraction of velocity *kept* per
# second (0.85 means 15% of speed is lost each second). Low, isotropic — most
# locomotion comes from the anisotropic leg drag below, not this.
FLUID_DAMPING = 0.85

# Anisotropic drag on each leg, the source of swimming thrust. Drag across the
# leg's broad face (perpendicular) is far stronger than along its length, so a
# leg sweeping sideways pushes the creature forward (a paddle/oar). Without this
# asymmetry, reciprocal leg motion produces zero net motion (scallop theorem).
LEG_DRAG_PERP = 14.0
LEG_DRAG_ALONG = 1.0
# Mild drag on the body hub so momentum bleeds off realistically.
BODY_DRAG = 3.0

# ---------------------------------------------------------------------------
# Creature defaults (a single morphology for the Phase-1 vertical slice)
# ---------------------------------------------------------------------------
DEFAULT_BODY_RADIUS = 22.0
DEFAULT_NUM_LEGS = 3
DEFAULT_LEG_LENGTH = 26.0
DEFAULT_LEG_WIDTH = 6.0
# Maximum motor speed (radians/sec) the brain can command per leg.
LEG_MAX_RATE = 6.0
# How far a leg may swing from its neutral angle (radians).
LEG_SWING_LIMIT = 0.9
# More legs -> a more elongated (longer) body. Each leg past the second adds
# this fraction to the body's length along its heading.
BODY_ELONGATION_PER_LEG = 0.18

# ---------------------------------------------------------------------------
# Vision (a finite, invisible cone in front of the creature)
# ---------------------------------------------------------------------------
import math as _math
# How far the creature can see (sim units). Targets beyond this are unseen.
VISION_RANGE = 420.0
# Half-angle of the forward vision cone (radians). ~60° each side = 120° cone.
VISION_HALF_ANGLE = _math.radians(60.0)

# ---------------------------------------------------------------------------
# Smell / chemical field
# ---------------------------------------------------------------------------
CHEM_GRID_COLS = 60
CHEM_GRID_ROWS = 60
CHEM_DIFFUSE = 0.18        # how fast smell spreads each step (0..1)
CHEM_DECAY = 0.995         # multiplicative fade each step
FOOD_SMELL_STRENGTH = 1.0  # concentration injected at a food cell per step
NUM_NOSTRILS = 4           # smell sample points around the body

# ---------------------------------------------------------------------------
# Energy / eating
# ---------------------------------------------------------------------------
START_ENERGY = 1.0
ENERGY_PER_FOOD = 0.6
FOOD_EAT_RADIUS = 18.0     # body-to-food distance that counts as "eaten"
CATCH_RADIUS = 30.0        # predator-to-prey distance that counts as a "catch"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPISODE_MAX_STEPS = 1000
# Named presets so the user never types a number. Maps label -> timesteps.
TRAIN_PRESETS = {
    "Quick": 50_000,
    "Normal": 200_000,
    "Thorough": 600_000,
}
DEFAULT_PRESET = "Quick"
