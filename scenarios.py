# scenarios.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
from kesslergame.scenario import Scenario

# =============================================================================
# Frozen-random layout utilities
# =============================================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0x7FFFFFFF)

def _sample_asteroid_states(
    rng: np.random.Generator,
    num_asteroids: int,
    map_size: Tuple[int, int],
    avoid_points: Optional[List[Tuple[float, float]]] = None,
    min_clear_radius: float = 60.0,
) -> List[Dict]:
    """
    Create randomized asteroid states with optional clear zone near the ship.
    """
    W, H = map_size
    out: List[Dict] = []
    avoid_points = avoid_points or []

    def is_clear(x: float, y: float) -> bool:
        for (ax, ay) in avoid_points:
            if (x - ax) ** 2 + (y - ay) ** 2 < (min_clear_radius ** 2):
                return False
        return True

    for _ in range(num_asteroids):
        for _attempt in range(100):
            x = float(rng.uniform(0, W))
            y = float(rng.uniform(0, H))
            if is_clear(x, y):
                break
        angle = float(rng.uniform(0.0, 360.0))
        speed = float(rng.uniform(40.0, 180.0))
        size = 4
        out.append({
            "position": (x, y),
            "angle": angle,
            "speed": speed,
            "size": size,
        })
    return out

def make_frozen_random_scenario(
    name: str,
    seed: int,
    num_asteroids: int = 7,
    map_size: Tuple[int, int] = (1000, 800),
    ship_states: Optional[List[Dict]] = None,
    time_limit: float = 60.0,
    ammo_limit_multiplier: float = 0.0,
    stop_if_no_ammo: bool = False,
) -> Scenario:
    """
    Creates a deterministic random asteroid layout for a given seed.
    """
    rng = _rng(seed)
    if ship_states is None:
        ship_states = [{"position": (400, 400), "angle": 90, "lives": 3, "team": 1}]
    avoid_points = [tuple(s.get("position", (400.0, 400.0))) for s in ship_states]
    asteroid_states = _sample_asteroid_states(rng, num_asteroids, map_size, avoid_points=avoid_points)

    return Scenario(
        name=name,
        asteroid_states=asteroid_states,
        ship_states=ship_states,
        map_size=map_size,
        time_limit=time_limit,
        ammo_limit_multiplier=ammo_limit_multiplier,
        stop_if_no_ammo=stop_if_no_ammo,
    )

class FrozenRandomManager:
    """
    Manages frozen random maps so all chromosomes in a generation
    see the same asteroid layout, while each generation uses a new one.
    """
    def __init__(self, base_seed: Optional[int] = None):
        if base_seed is None:
            env_val = os.environ.get("FROZEN_LAYOUT_BASE_SEED", "")
            try:
                base_seed = int(env_val) if env_val else 1337
            except Exception:
                base_seed = 1337
        self.base_seed = int(base_seed)
        self._cache: Dict[Tuple[int, int, int, Tuple[int, int]], Scenario] = {}

    def get(
        self,
        gen_idx: int,
        map_idx: int = 0,
        num_asteroids: int = 7,
        map_size: Tuple[int, int] = (1000, 800),
        ship_states: Optional[List[Dict]] = None,
        time_limit: float = 60.0,
        ammo_limit_multiplier: float = 0.0,
        stop_if_no_ammo: bool = False,
        name_prefix: str = "random_repeatable_frozen",
    ) -> Scenario:
        if ship_states is None:
            ship_states = [{"position": (400, 400), "angle": 90, "lives": 3, "team": 1}]

        key = (int(gen_idx), int(map_idx), int(num_asteroids), tuple(map_size))
        if key in self._cache:
            return self._cache[key]

        seed = (self.base_seed * 100003 + gen_idx * 1009 + map_idx * 17) & 0x7FFFFFFF
        scn = make_frozen_random_scenario(
            name=f"{name_prefix}_g{gen_idx}_m{map_idx}",
            seed=seed,
            num_asteroids=num_asteroids,
            map_size=map_size,
            ship_states=ship_states,
            time_limit=time_limit,
            ammo_limit_multiplier=ammo_limit_multiplier,
            stop_if_no_ammo=stop_if_no_ammo,
        )
        self._cache[key] = scn
        return scn

# Singleton for import use
FROZEN_RANDOM = FrozenRandomManager()

def random_repeatable_frozen(
    gen_idx: int,
    map_idx: int = 0,
    *,
    num_asteroids: int = 7,
    map_size: Tuple[int, int] = (1000, 800),
    ship_states: Optional[List[Dict]] = None,
    time_limit: float = 60.0,
    ammo_limit_multiplier: float = 0.0,
    stop_if_no_ammo: bool = False,
) -> Scenario:
    """Public helper: same map for all chromosomes in gen_idx, different between gens."""
    return FROZEN_RANDOM.get(
        gen_idx=gen_idx,
        map_idx=map_idx,
        num_asteroids=num_asteroids,
        map_size=map_size,
        ship_states=ship_states,
        time_limit=time_limit,
        ammo_limit_multiplier=ammo_limit_multiplier,
        stop_if_no_ammo=stop_if_no_ammo,
        name_prefix="random_repeatable_frozen",
    )


# =============================================================================
# Static scenarios
# =============================================================================

scenarios: Dict[str, Scenario] = {
    "random_repeatable": Scenario(
        name="random_repeatable",
        num_asteroids=7,
        asteroid_states=None,
        ship_states=[{"position": (400, 400), "angle": 90, "lives": 3, "team": 1}],
        map_size=(1000, 800),
        time_limit=60,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "battle_arena": Scenario(
        name="battle_arena",
        num_asteroids=15,
        ship_states=[{"position": (200, 200), "angle": 45, "lives": 5, "team": 2}],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.5,
        stop_if_no_ammo=True,
    ),
    "one_asteroid": Scenario(
        name="one_asteroid",
        ship_states=[{"position": (250, 200), "angle": 0, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (200, 200), "angle": 0.0, "speed": 0.001, "size": 1}
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "collision_test": Scenario(
        name="collision_test",
        ship_states=[{"position": (900, 225), "angle": 270, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (200, 200), "angle": 0.0, "speed": 150, "size": 4}
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "avoid": Scenario(
        name="avoid",
        ship_states=[{"position": (600, 200), "angle": 0, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (400, 200), "angle": 0.0, "speed": 150, "size": 3},
            {"position": (700, 200), "angle": 0.0, "speed": 150, "size": 3},
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "aim_trainer": Scenario(
        name="aim_trainer",
        ship_states=[{"position": (600, 500), "angle": 359, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (200, 200), "angle": 0, "speed": 150, "size": 4},
            {"position": (500, 200), "angle": 0, "speed": 150, "size": 4},
            {"position": (800, 200), "angle": 0, "speed": 150, "size": 4},
        ],
        map_size=(1200, 900),
        time_limit=90,
        ammo_limit_multiplier=0.0,
        stop_if_no_ammo=False,
    ),
    "crush": Scenario(
        name="crush",
        asteroid_states=[
            {"position": (200, 200), "angle": 50.0, "speed": 40, "size": 4},
            {"position": (300, 200), "angle": 60.0, "speed": 40, "size": 4},
            {"position": (400, 200), "angle": 70.0, "speed": 40, "size": 4},
            {"position": (500, 200), "angle": 90.0, "speed": 40, "size": 4},
        ],
        ship_states=[{"position": (500, 500), "angle": 0, "lives": 5, "team": 1}],
    ),
    "training1": Scenario(
        name="training1",
        ship_states=[{"position": (400, 400), "angle": 90, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (100, 100), "angle": 10, "speed": 100, "size": 4},
            {"position": (200, 100), "angle": 20, "speed": 100, "size": 4},
            {"position": (300, 100), "angle": 30, "speed": 100, "size": 4},
            {"position": (400, 100), "angle": 40, "speed": 100, "size": 4},
            {"position": (500, 100), "angle": 50, "speed": 100, "size": 4},
            {"position": (600, 100), "angle": 60, "speed": 100, "size": 4},
            {"position": (700, 100), "angle": 70, "speed": 100, "size": 4},
            {"position": (800, 100), "angle": 80, "speed": 100, "size": 4},
            {"position": (100, 700), "angle": 90, "speed": 100, "size": 4},
            {"position": (200, 700), "angle": 100, "speed": 100, "size": 4},
            {"position": (300, 700), "angle": 110, "speed": 100, "size": 4},
            {"position": (400, 700), "angle": 120, "speed": 100, "size": 4},
            {"position": (500, 700), "angle": 130, "speed": 100, "size": 4},
            {"position": (600, 700), "angle": 140, "speed": 100, "size": 4},
            {"position": (700, 700), "angle": 150, "speed": 100, "size": 4},
            {"position": (800, 700), "angle": 160, "speed": 100, "size": 4},
        ],
    ),
    "training2": Scenario(
        name="training2",
        ship_states=[{"position": (400, 400), "angle": 90, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (100, 100), "angle": 0, "speed": 100, "size": 4},
            {"position": (100, 200), "angle": 0, "speed": 100, "size": 4},
            {"position": (100, 300), "angle": 0, "speed": 100, "size": 4},
            {"position": (100, 400), "angle": 0, "speed": 100, "size": 4},
            {"position": (100, 500), "angle": 0, "speed": 100, "size": 4},
            {"position": (100, 600), "angle": 0, "speed": 100, "size": 4},
            {"position": (100, 700), "angle": 0, "speed": 100, "size": 4},
        ],
    ),
    "training3": Scenario(
        name="training3",
        ship_states=[{"position": (400, 400), "angle": 90, "lives": 5, "team": 1}],
        asteroid_states=[
            {"position": (100, 100), "angle": 45, "speed": 100, "size": 4},
            {"position": (700, 700), "angle": 225, "speed": 100, "size": 4},
            {"position": (100, 700), "angle": 315, "speed": 100, "size": 4},
            {"position": (700, 100), "angle": 135, "speed": 100, "size": 4},
        ],
    ),
}


# Example frozen random map (useful for debugging)
scenarios["random_repeatable_frozen_example_gen0"] = random_repeatable_frozen(
    gen_idx=0, map_idx=0, num_asteroids=7, map_size=(1000, 800)
)
