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
        self, chromosome, ship_state: "ShipOwnState", game_state: "GameState"
    ) -> "ActionsReturn":
        
        EPS = 1e-6
        thrust = EPS
        turn_angle = EPS
        shoot = False

            # Start total runtime measurement.
        total_start = time.time()
        """The actions method for the fuzzy controller."""

        # === Optimization Changes ===
        # 1. Cache functions and time.time() locally to speed up attribute lookups in loops.
        # 2. Precompute asteroid distances once (using math.hypot) and store them to avoid
        #    recalculating the distance in both the thrust and turn loops.
        # 3. Combine related asteroid data (position, velocity, original position, distance)
        #    into one sorted list to reduce redundant computations.
        # 4. Use enumerate and local variables in loops for faster iteration.

        # Cache frequently used functions for speed.
        _hypot = math.hypot
        _time = time.time
        _heading_relative_angle = vm.heading_relative_angle
        _calculate_closure_rate = vm.calculate_closure_rate
        _tsk_inference_const = ft.tsk_inference_const

        # Parameters from GA
        if chromosome is None:
            chromosome = [
                0.2, 0.5,
                0.5, 0.4, 0, 0.5, 0.6, 1, 0.5, 0.4, 0,
                0.5, 0.5, 0.5, 0.5,
                0.1, 0.4, 1, 0, 0.2, 0.6, 0, 0.1, 0.2,
                0, 0.1, 0.5, 0.2, 0.3, 0.7, 0.3, 0.7, 1,
                1,
                0.2, 0.5,
                0.5, 0.4, 0, 0.5, 0.6, 1, 0.5, 0.4, 0,
                0.5, 0.5, 0.5, 0.5,
                0.1, 0.4, 1, 0, 0.2, 0.6, 0.2, 0.1, 0.2,
                0.2, 0.1, 0.5, 0.2, 0.3, 0.7, 0.3, 0.7, 1,
                1
            ]


        # Best Chromosome from GA: [0.5 0.5 0.6 0.1 0.  0.3 1.  0.6 0.8 0.  0.4 0.4 0.9 0.4 0.8 0.2 0.9 0.4 0.6 0.4 0.  0.6 0.3 0.7 0.8 0.8 0.4 0.1 0.6 0.3 0.5 0.2 0.4]
        # Thrust Parameters
        time_start = _time()

        az_centers = [chromosome[0]]
        thrust_distance_centers = [chromosome[1]]

        az_mfs = ft.build_triangles(az_centers)
        thrust_distance_mfs = ft.build_triangles(thrust_distance_centers)
        rule_constants_thrust = np.array(chromosome[2:11]).reshape(
            len(az_mfs), len(thrust_distance_mfs)
        )  # [az, distance]




        # Turn Parameters
        angle_centers = [chromosome[12]]
        closure_centers = [chromosome[13]]

        turn_1_centers = [chromosome[14]]
        turn_distance_centers = [chromosome[15]]

        angle_mfs = ft.build_triangles(angle_centers)
        closure_mfs = ft.build_triangles(closure_centers)

        turn_1_mfs = ft.build_triangles(turn_1_centers)
        turn_distance_mfs = ft.build_triangles(turn_distance_centers)

        rule_constants_turn_1 = np.array(chromosome[15:24]).reshape(
            len(angle_mfs), len(closure_mfs)
        )  # [angle, closure]
        rule_constants_turn_2 = np.array(chromosome[24:33]).reshape(
            len(turn_1_mfs), len(turn_distance_mfs)
        )  # [turn_1, distance]


        # Threat Parameters

        threat_distance_centers = [chromosome[33]]
        threat_closure_centers = [chromosome[34]]

        threat_distance_mfs = ft.build_triangles(threat_distance_centers)  # [threat_distance]
        threat_closure_mfs = ft.build_triangles(threat_closure_centers)  # [threat_closure]

        rule_constants_threat = np.array(chromosome[35:44]).reshape(
            len(threat_distance_mfs), len(threat_closure_mfs)
        )  # [threat_distance, threat_closure]

        threat_scalar = chromosome[45]


        # Threat Avoidance Parameters

        threat_avoidance_distance_centers = [chromosome[46]]  # Distance for threat avoidance
        threat_avoidance_az_centers = [chromosome[47]]  # Azimuth for threat avoidance

        threat_avoidance_distance_mfs = ft.build_triangles(threat_avoidance_distance_centers)  # [threat_avoidance_distance]
        threat_avoidance_az_mfs = ft.build_triangles(threat_avoidance_az_centers)  # [threat_avoidance_az]

        rule_constants_threat_avoidance = np.array(chromosome[48:57]).reshape(
            len(threat_avoidance_distance_mfs), len(threat_avoidance_az_mfs)
        )

        turn_scalar = chromosome[58]

        thrust = EPS

        # Get asteroid properties from the game state.
        asteroid_positions = [asteroid["position"] for asteroid in game_state["asteroids"]]
        asteroid_velocities = [asteroid["velocity"] for asteroid in game_state["asteroids"]]
        asteroid_radii = [asteroid["radius"] for asteroid in game_state["asteroids"]]


        # Compute relative positions in the ship frame.
        relative_positions = vm.game_to_ship_frame(
            ship_state["position"], asteroid_positions, game_state["map_size"]
        )

        # === Optimization: Precompute distances and combine asteroid data ===
        # Compute distances once and combine with velocities and original positions.
        asteroid_data = [
            (pos, vel, orig_pos, _hypot(*pos))
            for pos, vel, orig_pos in zip(relative_positions, asteroid_velocities, asteroid_positions)
        ]
        # Sort asteroids by distance.
        asteroid_data.sort(key=lambda data: data[3])
        # Unpack the sorted data.
        relative_positions_sorted, asteroid_velocities_sorted, asteroid_positions_sorted, distances_sorted = map(
            list, zip(*asteroid_data)
        )


        threat_sum = 0
        for i, pos in enumerate(relative_positions_sorted):

            asteroid_distance = distances_sorted[i]  # Use precomputed distance
            # Skip asteroids that are too far.
            if asteroid_distance > 300:
                break

            # Calculate relative heading and normalize.
            closure_rate = _calculate_closure_rate(
                ship_state["position"],  # Ship position
                ship_state["heading"],  # Ship heading
                ship_state["speed"],  # Ship speed
                pos,  # Asteroid position
                asteroid_velocities_sorted[i],  # Asteroid velocity
            )
            # Avoid division by zero edge cases.
            if closure_rate == 0 or closure_rate == 1:
                relative_heading = 0.99999

            # Normalize distance.
            distance_norm = min(50 / (asteroid_distance + 0.0001), 0.99999)

            # Compute thrust contribution from this asteroid.
            threat = _tsk_inference_const(
                closure_rate,
                distance_norm,
                threat_closure_mfs,
                threat_distance_mfs,
                rule_constants_threat,
            )

            threat_sum += threat

        threat_sum = threat_sum * threat_scalar







        # === Build Thrust FIS ===





        if threat_scalar < 10:



            for i, pos in enumerate(relative_positions_sorted):
                asteroid_distance = distances_sorted[i]  # Use precomputed distance
                # Skip asteroids that are too far.
                if asteroid_distance > 200:
                    break

                # Calculate relative heading and normalize.
                relative_heading = _heading_relative_angle([0, 0], ship_state["heading"], pos) / 360
                # Avoid division by zero edge cases.
                if relative_heading == 0 or relative_heading == 1:
                    relative_heading = 0.99999

                # Normalize distance.
                distance_norm = min(50 / (asteroid_distance + 0.0001), 0.99999)

                # Compute thrust contribution from this asteroid.
                thrust_sum = _tsk_inference_const(
                    relative_heading,
                    distance_norm,
                    az_mfs,
                    thrust_distance_mfs,
                    rule_constants_thrust,
                ) - 0.5

                thrust += thrust_sum

            thrust = thrust * 2000 * chromosome[33]

            # === Build Turn FIS ===

            turn_asteroid_threat = -1
            turn_asteroid_index = 0

            total_fuzzy_time = EPS

            heading = ship_state["heading"]
            position = ship_state["position"]
            speed = ship_state["speed"]
            max_iterations = min(len(relative_positions_sorted), 15)

            # Localize function references.
            heading_relative_angle = _heading_relative_angle
            calculate_closure_rate = _calculate_closure_rate
            tsk_inference_const = _tsk_inference_const

            for i in range(max_iterations):
                pos = relative_positions_sorted[i]
                rel_heading = heading_relative_angle([0, 0], heading, pos) / 360

                asteroid_distance = distances_sorted[i]  # Use precomputed distance
                distance_norm = min(50 / (asteroid_distance + 0.0001), 0.99999)
                off_nose_norm = 1 - abs(1 - 2 * rel_heading)

                closure_rate = calculate_closure_rate(
                    position,
                    heading,
                    speed,
                    pos,
                    asteroid_velocities_sorted[i],
                )

                fuzzy_val_1 = tsk_inference_const(
                    off_nose_norm, closure_rate, angle_mfs, closure_mfs, rule_constants_turn_1
                )
                fuzzy_val_final = tsk_inference_const(
                    fuzzy_val_1, distance_norm, turn_1_mfs, turn_distance_mfs, rule_constants_turn_2
                )


                if fuzzy_val_final > turn_asteroid_threat:
                    turn_asteroid_index = i



            # Calculate turn angle using the most threatening asteroid.
            turn_angle, on_target = vm.turn_angle(
                ship_state["position"],
                ship_state["heading"],
                ship_state["turn_rate_range"],
                self.bullet_speed,
                asteroid_positions_sorted[turn_asteroid_index],
                asteroid_velocities_sorted[turn_asteroid_index],
                game_state["delta_time"],
            )

            # Determine if we should shoot.
            if on_target:
                shoot = True

        else:

            
            for i, pos in enumerate(relative_positions_sorted):
                asteroid_distance = distances_sorted[i]  # Use precomputed distance
                # Skip asteroids that are too far.
                if asteroid_distance > 200:
                    break

                # Calculate relative heading and normalize.
                relative_heading = _heading_relative_angle([0, 0], ship_state["heading"], pos) / 360
                # Avoid division by zero edge cases.
                if relative_heading == 0 or relative_heading == 1:
                    relative_heading = 0.99999

                # Normalize distance.
                distance_norm = min(50 / (asteroid_distance + 0.0001), 0.99999)

                # Compute thrust contribution from this asteroid.
                threat_avoidance = _tsk_inference_const(
                    relative_heading,
                    distance_norm,
                    az_mfs,
                    thrust_distance_mfs,
                    rule_constants_thrust,
                )

                threat_avoidance_sum = threat_avoidance

            turn_angle = threat_avoidance_sum * turn_scalar * 500


            
        return thrust, turn_angle, shoot, False
