from kesslergame.kessler_game import KesslerGame
from kesslergame.scenario import Scenario
from src.controller import FuzzyController


if __name__ == "__main__":
    # Available settings
    settings = {
        "frequency": 20,
        "real_time_multiplier": 2,
        "graphics_on": True,
        "sound_on": False,
        "prints": True,
    }

    # Instantiate an instance of FuzzyAsteroidGame
    game = KesslerGame(
        settings=settings,
        track_compute_cost=True,
        controller_timeout=True,
        ignore_exceptions=False,
    )

    scenario_ship = Scenario(
        name="Test",
        num_asteroids=3,
        ship_states=[{"position": (300, 500), "angle": 180, "lives": 1}],
    )

    score = game.run(
        controller=dict({1: FuzzyController(), 2: FuzzyController()}),
        scenario=scenario_ship,
    )
