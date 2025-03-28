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

        pos_vel_pairs = sorted(zip(relative_positions, asteroid_velocities), key=lambda pv: math.hypot(*pv[0]))
        relative_positions_sorted, asteroid_velocities_sorted = map(list, zip(*pos_vel_pairs))

        # Build Thrust FIS

        az_centers = [0.5]
        distance_centers = [0.5]


        # Build membership functions for x1 and x2 based on the provided centers.
        az_mfs = ft.build_triangles(az_centers)
        distance_mfs = ft.build_triangles(distance_centers)

        # ft.plot_mfs(distance_mfs)
     
        rule_constants_thrust = np.array([0,-100,-500,0,100,500,0,-100,-500]).reshape(len(az_mfs), len(distance_mfs)) # [az, distance]
        thrust = 0

        print("\nStart Thrust FIS Loop")
        for i in range(len(relative_positions_sorted)):
            

            # Check asteroid is not so far that we do not care about it
            asteroid_distance = math.hypot(*relative_positions_sorted[i])
            if asteroid_distance>500:
                break

            # Calculate relative heading
            relative_heading = vm.heading_relative_angle(
                [0,0],
                ship_state["heading"],
                relative_positions_sorted[i],
            ) / 360


            # If relative heading is 0 or 1, set it to 0.99999 to avoid division by zero
            if relative_heading == 0 or relative_heading == 1:
                relative_heading = 0.99999

            # Calculate thrust per asteroid
            distance_norm = min(50/(asteroid_distance+0.0001),0.99999)
            thrust_sum = ft.tsk_inference_const(relative_heading, distance_norm, az_mfs, distance_mfs, rule_constants_thrust)
            thrust += thrust_sum

        thrust = thrust * 200

        # Build Turn FIS

        angle_centers = [0.5]
        distance_centers = [0.5]

        turn_1_centers = [0.5]
        closure_centers = [0.5]


        # Build membership functions for x1 and x2 based on the provided centers.

        angle_mfs = ft.build_triangles(angle_centers)
        distance_mfs = ft.build_triangles(distance_centers)

        turn_1_mfs = ft.build_triangles(turn_1_centers)
        closure_mfs = ft.build_triangles(closure_centers)

        # Build Rules List

        rule_constants_turn_1 = np.array([0.1,0.4,1,0,0.2,0.6,0,0.1,0.2]).reshape(len(angle_mfs), len(distance_mfs)) 
        # [angle, distance] Angle closer to 0 is closer to off the nose, Distance closer to 1 is closer to ship


        rule_constants_turn_2 = np.array([0,0.1,0.4,0.0,0.3,0.7,0.1,0.4,1]).reshape(len(turn_1_mfs), len(closure_mfs)) # [turn_1, closure]
        # [turn_1, closure] Turn 1 closer to 1 is more threat, Closure closer to 1 is more threat


        turn_angle = 0
        turn_asteroid_threat = -1
        turn_asteroid_index = 0
        
        for i in range(len(relative_positions_sorted)):

            # Calculate relative heading
            relative_heading = vm.heading_relative_angle(
                [0,0],
                ship_state["heading"],
                relative_positions_sorted[i],
            ) / 360

            asteroid_distance = math.hypot(*relative_positions_sorted[i])
            distance_norm = min(50/(asteroid_distance+0.0001),0.99999)
            
            off_nose_norm = 1 - abs(1 - 2*relative_heading)


            turn_fis_val_1 = ft.tsk_inference_const(off_nose_norm, distance_norm, angle_mfs, distance_mfs, rule_constants_turn_1)

            closure_rate = vm.calculate_closure_rate(
                ship_state["position"],
                ship_state["heading"],
                ship_state["speed"],
                relative_positions_sorted[i],
                asteroid_velocities_sorted[i],
            )

            turn_fis_val_final = ft.tsk_inference_const(turn_fis_val_1, closure_rate, turn_1_mfs, closure_mfs, rule_constants_turn_2)

            if turn_fis_val_final > turn_asteroid_threat:
                turn_asteroid_threat = turn_fis_val_final
                turn_asteroid_index = i

        turn_angle = vm.

            
            











        return thrust, turn_angle, shoot, False
