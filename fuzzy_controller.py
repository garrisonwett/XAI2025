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



        # Convert from game relative to ship relative
        relative_positions = vm.game_to_ship_frame(ship_state["position"], asteroid_positions, game_state["map_size"])
        closest_asteroid_distance = 1000000
        asteroids_in_distance = 0


        
        for i in range(len(relative_positions)):
            distance_to_asteroid = vm.distance_to(relative_positions[i])
            if distance_to_asteroid < 300:
                asteroids_in_distance += 1
            if distance_to_asteroid < closest_asteroid_distance:
                closest_asteroid_distance = distance_to_asteroid
                closest_asteroid_index = i


        closest_asteroid_position = relative_positions[closest_asteroid_index]

        relative_positions_sorted_index = vm.sort_by_distance(relative_positions)
        
        relative_positions_sorted = []
        for i in range(len(relative_positions_sorted_index)):
            relative_positions_sorted.append(vm.distance_to(relative_positions[relative_positions_sorted_index[i]]))

        turn_angle, on_target = vm.turn_angle(
            ship_state["position"],
            ship_state["heading"],
            ship_state["turn_rate_range"],
            self.bullet_speed,
            asteroid_positions[closest_asteroid_index],
            asteroid_velocities[closest_asteroid_index],
            game_state["delta_time"],
        )

        collide, collide_time = vm.calculate_if_collide(
            ship_state["position"],
            ship_state["heading"],
            ship_state["speed"],
            ship_state["radius"],
            relative_positions[0],
            asteroid_velocities[0],
            asteroid_radii[0],
        )

        relative_heading = vm.heading_relative_angle(
            [0,0],
            ship_state["heading"],
            closest_asteroid_position,
        )
        
        closure_rate = vm.calculate_closure_rate(
            ship_state["position"],
            ship_state["heading"],
            ship_state["speed"],
            relative_positions[closest_asteroid_index],
            asteroid_velocities[closest_asteroid_index],
        )
    



        # Define your "middle" centers 
        az_centers = [0.5] 
        closure_centers = [0.5]

        # Build membership functions for x1 and x2 based on the provided centers.
        az_mfs = ft.build_triangles(az_centers)
        closure_mfs = ft.build_triangles(closure_centers)

        num_rules_az = len(az_mfs)
        num_rules_closure = len(closure_mfs)


        # Normalize Inputs

        relative_heading = relative_heading / 360
        closure_rate = 1


        if relative_heading < 0.0001:
            relative_heading = 0.001
        if relative_heading > 0.9999:
            relative_heading = 0.9999
        if closure_rate < 0.0001:
            closure_rate = 0.001
        if closure_rate > 0.9999:
            closure_rate = 0.9999

        def f(x1, p1):
            """
            Computes f(x1) = -1 * (p1 * abs(x1 - 0.5) - 0.25)
            in a piecewise manner, explicitly factoring out p1*x1.
            """
            if x1 < 0.5:
                return p1 + (0.25 - 0.5 * p1)
            else:
                return -p1 + (0.25 + 0.5 * p1)

        thrust_fis_1_params = []


        # Generate the parameters for the TSK system
        for i in range(num_rules_az):
            row = []
            for j in range(num_rules_closure):
                p1 = f(relative_heading, 1)
                p2 = max(j-1,0)
                row.append([p1, p2])
            thrust_fis_1_params.append(row)


        sorted_len = len(relative_positions_sorted)

        thrust_sum = 0
        for i in range(min(asteroids_in_distance,sorted_len)):
            distance = relative_positions_sorted[i]
            distance_norm = math.sqrt(min(50/(distance+0.0001),0.99999))
            thrust_sum = distance_norm * ft.tsk_inference_mult(x1=relative_heading, x2=closure_rate, x1_mfs=az_mfs, x2_mfs=closure_mfs, params=thrust_fis_1_params)
            thrust_sum += thrust_sum
        print(thrust_sum)
        thrust = thrust_sum * 700


        shoot=False
        if on_target == True:
            shoot = True

        return thrust, turn_angle, shoot, False
