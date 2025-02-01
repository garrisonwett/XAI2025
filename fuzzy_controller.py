from typing import TYPE_CHECKING

from kesslergame import KesslerController

from utils import LoggerUtility
from utils.kessler_helpers import get_bullet_speed
from utils.math import vector_math as vm
from fuzzy_logic.fuzzy_trees import tsk_inference, build_triangles
if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState
import time

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



        
        relative_positions = vm.game_to_ship_frame(ship_state["position"], asteroid_positions, game_state["map_size"])

        closest_asteroid_distance = 1000000
        for i in range(len(relative_positions)):
            distance_to_asteroid = vm.distance_to(relative_positions[i])
            if distance_to_asteroid < closest_asteroid_distance:
                closest_asteroid_distance = distance_to_asteroid
                closest_asteroid_index = i
        closest_asteroid_position = relative_positions[closest_asteroid_index]

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
    
        # Calculate Fuzzy Computation Time

        start_time = time.time()

    # Define your "middle" centers (they don't need to be fixed in number)
        az_centers = [0.5] 
        closure_centers = [0.5]
        

        # You could also try: centers = [0.2, 0.4, 0.6, 0.9] (which yields 6 MFs)

        # Build membership functions for x1 and x2 based on the provided centers.
        x1_mfs = build_triangles(az_centers)
        x2_mfs = build_triangles(closure_centers)

        # Visualize the membership functions
        # plot_mfs(x1_mfs, x_range=(0,1), title="x1 Membership Functions")
        # plot_mfs(x2_mfs, x_range=(0,1), title="x2 Membership Functions")

        # Set up realistic parameters for the TSK rule consequents.
        # For each rule, we assume a linear consequent: y = p0 + p1*x1 + p2*x2.
        # Since the number of rules equals len(x1_mfs) x len(x2_mfs),
        # we create a parameter matrix accordingly.
        num_rules_x1 = len(x1_mfs)
        num_rules_x2 = len(x2_mfs)
        params = []
        for i in range(num_rules_x1):

            row = []
            for j in range(num_rules_x2):
                # Example: p0, p1, and p2 are chosen based on the rule indices.
                p1 = 1 
                p2 = max(j-1,0)
                row.append([p1, p2])
            params.append(row)

        # Normalize Inputs
        print("\n")
        print("--------------------------------------------")

        print(turn_angle)
        print(relative_heading, closure_rate)
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
        print(relative_heading, closure_rate)
        thrust = tsk_inference(x1=relative_heading, x2=closure_rate, x1_mfs=x1_mfs, x2_mfs=x2_mfs, params=params)
        print(thrust)
        end_time = time.time()
        computation_time = end_time - start_time


        shoot=False
        if on_target == True:
            shoot = True

        return thrust, turn_angle, shoot, False
