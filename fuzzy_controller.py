from typing import TYPE_CHECKING

from kesslergame import KesslerController

from utils import LoggerUtility
from utils.kessler_helpers import get_bullet_speed
from utils.math import vector_math as vm
from fuzzy_logic.fuzzy_trees import thrust_tree
if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState


class FuzzyController(KesslerController):
    """
    The main class for the fuzzy controller.

    Args:
       KesslerController (KesslerController): The base class for the controller. Provided by the kesslergame package.
    """

    def __init__(self):
        """
        Initializes the fuzzy controller.
        All state variables should be initialized here.
        """
        super().__init__()

        self._name: str = "BajaBlasteroids"

        # Ship Variables

        # Bullet Variables
        self.bullet_speed = get_bullet_speed()

        # Threat Variables

    @property
    def name(self) -> str:
        """Getter method for the name of the controller."""
        return self._name

    def control(self, observation):
        # Implement your fuzzy logic here
        return 0

    def explanation(self) -> str:
        # Just returns the most recent message. Ideally they would call this whenever self.msg is updated
        return self.msg

    def actions(
        self, ship_state: "ShipOwnState", game_state: "GameState"
    ) -> "ActionsReturn":
        """The actions method for the fuzzy controller."""

        asteroid_positions = [
            asteroid["position"] for asteroid in game_state["asteroids"]
        ]
        asteroid_velocities = [
            asteroid["velocity"] for asteroid in game_state["asteroids"]
        ]
        asteroid_radii = [
            asteroid["radius"] for asteroid in game_state["asteroids"]
        ]

        turn_angle = vm.turn_angle(
            ship_state["position"],
            ship_state["heading"],
            ship_state["turn_rate_range"],
            self.bullet_speed,
            asteroid_positions[0],
            asteroid_velocities[0],
            game_state["delta_time"],
        )

        collide, collide_time = vm.calculate_if_collide(
            ship_state["position"],
            ship_state["heading"],
            ship_state["speed"],
            ship_state["radius"],
            asteroid_positions[0],
            asteroid_velocities[0],
            asteroid_radii[0],
        )

        relative_heading = vm.heading_relative_angle(
            ship_state["position"],
            ship_state["heading"],
            asteroid_positions[0],
        )

        closure_rate = vm.calculate_closure_rate(
            ship_state["position"],
            ship_state["heading"],
            ship_state["speed"],
            asteroid_positions[0],
            asteroid_velocities[0],
        )

        
        thrust = thrust_tree(closure_rate, relative_heading)
        print(thrust)
        return thrust, turn_angle, False, True
