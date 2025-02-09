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

    # def control(self, observation):
    #     # Implement your fuzzy logic here
    #     return 0

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
        closest_asteroid_size = relative_positions[closest_asteroid_index]

        relative_positions_sorted_index = vm.sort_by_distance(relative_positions)
        
        relative_positions_sorted = []
        asteroid_radii_sorted = []
        for i in range(len(relative_positions_sorted_index)):
            relative_positions_sorted.append(vm.distance_to(relative_positions[relative_positions_sorted_index[i]]))
            asteroid_radii_sorted.append(asteroid_radii[relative_positions_sorted_index[i]])
        
        # turn_angle, on_target = vm.turn_angle(
        #     ship_state["position"],
        #     ship_state["heading"],
        #     ship_state["turn_rate_range"],
        #     self.bullet_speed,
        #     asteroid_positions[closest_asteroid_index], 
        #     asteroid_velocities[closest_asteroid_index],
        #     game_state["delta_time"],
        # )

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
        size_centers = [0.5]
        distance_centers = [0.5]

        # Build membership functions for x1 and x2 based on the provided centers.
        az_mfs = ft.build_triangles(az_centers)
        closure_mfs = ft.build_triangles(closure_centers)
        size_mfs = ft.build_triangles(size_centers)
        distance_mfs = ft.build_triangles(distance_centers)

        # ft.plot_mfs(distance_mfs)

        num_rules_az = len(az_mfs)
        num_rules_closure = len(closure_mfs)
        num_rules_size = len(size_mfs)
        num_rules_distance = len(distance_mfs)


        # Normalize Inputs

        relative_heading = relative_heading / 360
        closure_rate = 1
        for i in range(len(asteroid_radii)):
            asteroid_radii[i] = asteroid_radii[i] / 32
        


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


        # Generate the parameters for the TSK system - thrust and aim 1.1
        for i in range(num_rules_az):
            row = []
            for j in range(num_rules_closure):
                p1 = f(relative_heading, 1)
                p2 = f(closure_rate,1)
                row.append([p1, p2])
                # print(thrust_fis_1_params)
            thrust_fis_1_params.append(row)

        aim_fis_1_2_params = []
        # Generate the parameters for the TSK system - Aim 1.2 
        for i in range(num_rules_size):
            row = []
            for j in range(num_rules_distance):
                p1 = f(asteroid_radii[i]/len(asteroid_radii), 1)
                p2 = f(relative_positions_sorted[i]/math.sqrt(min(50/(relative_positions_sorted[i]+0.0001),0.99999)),1)
                row.append([p1, p2])
                
            aim_fis_1_2_params.append(row)
        print(aim_fis_1_2_params)
        
        aim_fis_2_params = []
        # Generate the parameters for the TSK system - Aim 2 
        for i in range(num_rules_az):
            row = []
            for j in range(num_rules_closure):
                p1 = f(asteroid_radii[i]/len(asteroid_radii), 1)
                p2 = max(j-1,0)
                row.append([p1, p2])
                
            aim_fis_2_params.append(row)


        sorted_len = len(relative_positions_sorted)

        # Thrust FIS
        thrust_sum = 0
        for i in range(min(asteroids_in_distance,sorted_len)):
            distance = relative_positions_sorted[i]
            distance_norm = math.sqrt(min(50/(distance+0.0001),0.99999))
            thrust_sum = distance_norm * ft.tsk_inference_mult(x1=relative_heading, x2=closure_rate, x1_mfs=az_mfs, x2_mfs=closure_mfs, params=thrust_fis_1_params)
            thrust_sum += thrust_sum
        # ft.plot_tsk_surface(x1_mfs=az_mfs, x2_mfs=closure_mfs, params=thrust_fis_1_params, resolution=50)
        thrust = thrust_sum * 700
        # thrust = 0
        print(thrust)
        

        # Aim FIS 1.1 (in: Heading [deg.], Closure rate | out: threat level[-1,1])

        threat_sum = 0
        for i in range(sorted_len):
            distance = relative_positions_sorted[i]
            distance_norm = math.sqrt(min(50/(distance+0.0001),0.99999))
            # print(relative_heading)
            threat_sum = distance_norm * ft.tsk_inference_mult(x1=relative_heading, x2=closure_rate, x1_mfs=az_mfs, x2_mfs=closure_mfs, params=thrust_fis_1_params)
            threat_sum += threat_sum
        
        threat = threat_sum
        


        # Aim FIS 1.2 (in: Asteroid Size, Distance | out: hit chance)
        
        aim_sum = 0
        for i in range(sorted_len):
            distance = relative_positions_sorted[i]
            distance_norm = math.sqrt(min(50/(distance+0.0001),0.99999))
            # print(asteroid_radii[i])
            aim_sum =  distance_norm* ft.tsk_inference_mult(x1=asteroid_radii_sorted[i], x2=distance_norm, x1_mfs=size_mfs, x2_mfs=distance_mfs, params=aim_fis_1_2_params)
            aim_sum += aim_sum
        aim = aim_sum
        # ft.plot_tsk_surface(x1_mfs=size_mfs, x2_mfs=distance_mfs, params=aim_fis_1_2_params, resolution=50)
        # print(aim)
        

        # Aim FIS 2
        best_asteroid_index = -1  # Default value if no asteroid is found
        best_aim_score = float('-inf')  # Initialize with a very low value
        for i in range(sorted_len):
            # Compute fuzzy inference output (aim score)
            aim_score = ft.tsk_inference_mult(
                x1=aim, x2=threat, 
                x1_mfs=size_mfs, x2_mfs=distance_mfs, 
                params=aim_fis_2_params)

            # Check if this asteroid has the highest aim score
            if aim_score > best_aim_score:
                best_aim_score = aim_score
                best_asteroid_index = i  # Store the index of the best asteroid

        # The output is now the index of the chosen asteroid
        chosen_asteroid_index = best_asteroid_index
        print(chosen_asteroid_index)

        turn_angle, on_target = vm.turn_angle(
            ship_state["position"],
            ship_state["heading"],
            ship_state["turn_rate_range"],
            self.bullet_speed,
            asteroid_positions[chosen_asteroid_index], 
            asteroid_velocities[chosen_asteroid_index],
            game_state["delta_time"],
        )


    
        shoot=False
        if on_target == True:
            shoot = True

        return thrust, turn_angle, shoot, False
