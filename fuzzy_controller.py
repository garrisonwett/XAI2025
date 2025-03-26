from typing import TYPE_CHECKING

from kesslergame import KesslerController

from utils import LoggerUtility
from utils.kessler_helpers import get_bullet_speed
from utils.math import vector_math as vm
from fuzzy_logic import fuzzy_trees as ft
if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState
import time
import math
import numpy as np



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

    # def control(self, observation):
    #     # Implement your fuzzy logic here
    #     return 0

    def explanation(self) -> str:
        # Just returns the most recent message. Ideally they would call this whenever self.msg is updated
        return self.msg

    def actions(
        self, ship_state: "ShipOwnState", game_state: "GameState"
    ) -> "ActionsReturn":
        thrust = 0
        turn_angle = 0
        shoot = False
        
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

        EPS = 1e-9

        relative_positions = vm.game_to_ship_frame(ship_state["position"], asteroid_positions, game_state["map_size"])

        # Sort positions by distance (√(dx²+dy²))
        relative_positions_sorted = sorted(
            relative_positions,
            key=lambda pos: math.hypot(*pos)
        )
        # Build Thrust FIS

        az_centers = [0.5]
        distance_centers = [0.5]


        # Build membership functions for x1 and x2 based on the provided centers.
        az_mfs = ft.build_triangles(az_centers)
        distance_mfs = ft.build_triangles(distance_centers)

        # ft.plot_mfs(distance_mfs)
     
        rule_constants = np.array([0,-100,-500,0,100,500,0,-100,-500]).reshape(len(az_mfs), len(distance_mfs))
        thrust = 0

        print("Start FIS Loop")
        for i in range(len(relative_positions_sorted)):
            
            asteroid_distance = math.hypot(*relative_positions_sorted[i])
            if asteroid_distance>500:
                break


            relative_heading = vm.heading_relative_angle(
                [0,0],
                ship_state["heading"],
                relative_positions_sorted[i],
            ) / 360

            if relative_heading == 0 or relative_heading == 1:
                relative_heading = 0.99999

            distance_norm = min(50/(asteroid_distance+0.0001),0.99999)
            thrust_sum = ft.tsk_inference_const(relative_heading, distance_norm, az_mfs, distance_mfs, rule_constants)-0.5
            print(thrust_sum)
            thrust += thrust_sum
        thrust = thrust * 500
        print(thrust)


        return thrust, turn_angle, shoot, False
