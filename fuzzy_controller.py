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

        self.counter = 1
        self.mode = "Avoidance"

        # Ship Variables

        

        self.asteroids_shot_at = []




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
        if len(self.asteroids_shot_at) >= 20:
            self.asteroids_shot_at.pop(0) 
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
            chromosome = [0.85523172, 0.6853809, 0.99123998, 0.91451783, 0.53872943, 0.35664217, 0.25674431, 0.08803115, 0.18758472, 0.96246244, 0.1916734, 0.96833484, 0.02628331, 0.57785765, 0.1196252, 0.63225362, 0.46882679, 0.71777299, 0.82152798, 0.59765353, 0.15332517, 0.22691668, 0.67164351, 0.51825907, 0.60413527, 0.36230559, 0.13900076, 0.11432595, 0.71273771, 0.6605982, 0.45888102, 0.28467731, 0.86633552, 0.20717115, 0.49676342, 0.56947511, 0.33032694, 0.7745347, 0.38238305, 0.26854836, 0.70754135, 0.39565687, 0.67469258, 0.23880572, 0.37223091, 0.57504392, 0.73465904, 0.09994881, 0.27618423, 0.88286071, 0.28346158, 0.21135125, 0.19127665, 0.75932432, 0.08345862, 0.58844292, 0.36830784, 0.3586154, 0.31159994, 0.53552311, 0.29707806, 0.8697389, 0.24947505, 0.03634646, 0.75150587, 0.39691107, 0.41059989, 0.93108836]
        # Best Chromosome from GA: [0.5 0.5 0.6 0.1 0.  0.3 1.  0.6 0.8 0.  0.4 0.4 0.9 0.4 0.8 0.2 0.9 0.4 0.6 0.4 0.  0.6 0.3 0.7 0.8 0.8 0.4 0.1 0.6 0.3 0.5 0.2 0.4]
        # Thrust Parameters

        # Scalar Values
        threat_sum_scalar_1, chromosome = chromosome[0], chromosome[1:]
        thrust_sum_scalar_4, chromosome = chromosome[0], chromosome[1:]




        # FIS 1
        closure_centers_1, chromosome = chromosome[:1], chromosome[1:]
        distance_centers_1, chromosome = chromosome[:1], chromosome[1:]

        closure_mfs_1 = ft.build_triangles(closure_centers_1)
        distance_mfs_1 = ft.build_triangles(distance_centers_1)

        rule_constants_threat_1 = np.array(chromosome[:9]).reshape(
            len(closure_mfs_1), len(distance_mfs_1)
        )
        chromosome = chromosome[9:]


        # FIS 2
        relative_heading_centers_2, chromosome = chromosome[:1], chromosome[1:]
        size_centers_2, chromosome = chromosome[:1], chromosome[1:]

        relative_heading_mfs_2 = ft.build_triangles(relative_heading_centers_2)
        size_mfs_2 = ft.build_triangles(size_centers_2)

        rule_constants_threat_2 = np.array(chromosome[:9]).reshape(
            len(relative_heading_mfs_2), len(size_mfs_2)
        )
        chromosome = chromosome[9:]


        # FIS 3
        fis_centers_1_3, chromosome = chromosome[:1], chromosome[1:]
        fis_centers_2_3, chromosome = chromosome[:1], chromosome[1:]

        threat_fis_mfs_1 = ft.build_triangles(fis_centers_1_3)
        threat_fis_mfs_2 = ft.build_triangles(fis_centers_2_3)

        rule_constants_threat_3 = np.array(chromosome[:9]).reshape(
            len(threat_fis_mfs_1), len(threat_fis_mfs_2)
        )
        chromosome = chromosome[9:]


        # FIS 4
        az_centers_4, chromosome = chromosome[:1], chromosome[1:]
        thrust_distance_centers_4, chromosome = chromosome[:1], chromosome[1:]

        az_mfs_4 = ft.build_triangles(az_centers_4)
        thrust_distance_mfs_4 = ft.build_triangles(thrust_distance_centers_4)

        rule_constants_thrust_4 = np.array(chromosome[:9]).reshape(
            len(az_mfs_4), len(thrust_distance_mfs_4)
        )
        chromosome = chromosome[9:]


        # FIS 5
        az_centers_5, chromosome = chromosome[:1], chromosome[1:]
        distance_centers_5, chromosome = chromosome[:1], chromosome[1:]

        az_mfs_5 = ft.build_triangles(az_centers_5)
        distance_mfs_5 = ft.build_triangles(distance_centers_5)

        rule_constants_thrust_5 = np.array(chromosome[:9]).reshape(
            len(az_mfs_5), len(distance_mfs_5)
        )
        chromosome = chromosome[9:]


        # FIS 6

        relative_heading_centers_6, chromosome = chromosome[:1], chromosome[1:]
        fis_centers_1_6, chromosome = chromosome[:1], chromosome[1:]

        relative_heading_mfs_6 = ft.build_triangles(relative_heading_centers_6)
        defensive_fis_mfs_1_6 = ft.build_triangles(fis_centers_1_6)

        rule_constants_threat_6 = np.array(chromosome[:9]).reshape(
            len(relative_heading_mfs_6), len(defensive_fis_mfs_1_6)
        )
        chromosome = chromosome[9:]












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

        ship_respawining = False
        if ship_state["is_respawning"] == True:
            ship_respawining = True





        # Todo: 
        # Assign a threat value to each asteroid (FIS)
        # Determine which mode to operate in (threat avoidance or threat shooting) (FIS)
        # Create threat avoidance fis and determine to shoot or run (FIS)
        # Create threat shooting fis (FIS)
        # Mine or not (FIS)



        # Assign Asteroids Threat Values
        threat_array = []
        proximity_threat = 0
        for i, pos in enumerate(relative_positions_sorted):
 
            asteroid_distance = distances_sorted[i]  # Use precomputed distance
            distance_norm = min(50 / (asteroid_distance + EPS), 0.99999)


            closure_rate = _calculate_closure_rate(
                ship_state["position"],  # Ship position
                ship_state["heading"],  # Ship heading
                ship_state["speed"],  # Ship speed
                pos,  # Asteroid position
                asteroid_velocities_sorted[i],  # Asteroid velocity
            )
            closure_rate = min(max((closure_rate+200)/400, 0), 1)
            # Avoid division by zero edge cases.
            

            relative_heading = _heading_relative_angle([0, 0], ship_state["heading"], pos) / 360

            # Avoid division by zero edge cases.
            if relative_heading == 0 or relative_heading == 1:
                relative_heading = 0.99999

            size = asteroid_radii[i] / 4  # Normalize size to a range of 0-1

            threat_fis_1_output = _tsk_inference_const(
                closure_rate,
                distance_norm,
                closure_mfs_1,
                distance_mfs_1,
                rule_constants_threat_1,
            )

            threat_fis_2_output = _tsk_inference_const(
                relative_heading,
                size,
                relative_heading_mfs_2,
                size_mfs_2,
                rule_constants_threat_2,
            )   
            
            asteroid_threat = _tsk_inference_const(
                threat_fis_1_output,
                threat_fis_2_output,
                threat_fis_mfs_1,
                threat_fis_mfs_2,
                rule_constants_threat_3,
            )


            # Append the threat value to the list.
            threat_array.append(asteroid_threat)

            if asteroid_distance < 400:
                proximity_threat += asteroid_threat



        # Determine which mode to operate in (threat avoidance or threat shooting)    
        
        previous_mode = self.mode

        if ship_respawining == True or proximity_threat > 20 * threat_sum_scalar_1:
            self.mode = "Defensive"
        else:
            self.mode = "Offensive"

        # if self.mode != previous_mode:
        #     print(f"Mode changed to: {self.mode}")


        # Offensive Mode
        if self.mode == "Offensive":

            shot_set = set(self.asteroids_shot_at)

            # get all indices, sorted by threat strength (descending)
            sorted_idxs = np.argsort(threat_array)[::-1]

            # scan until you find one not yet shot
            threat_index = -1
            for idx in sorted_idxs:
                if idx not in shot_set:
                    threat_index = idx
                    self.asteroids_shot_at.append(threat_index)
                    break
            
            # Calculate turn angle using the most threatening asteroid.
            turn_angle, on_target = vm.turn_angle(
                ship_state["position"],
                ship_state["heading"],
                ship_state["turn_rate_range"],
                self.bullet_speed,
                asteroid_positions_sorted[threat_index],
                asteroid_velocities_sorted[threat_index],
                game_state["delta_time"],
            )

            # Determine if we should shoot.
            if on_target:
                self.asteroids_shot_at.append(threat_index)
                shoot = True


            # === Build Thrust FIS ===


            for i, pos in enumerate(relative_positions_sorted):
                asteroid_distance = distances_sorted[i]  # Use precomputed distance
                # Skip asteroids that are too far.
                if asteroid_distance > 300:
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
                    az_mfs_4,
                    thrust_distance_mfs_4,
                    rule_constants_thrust_4,
                ) - 0.5

                thrust += thrust_sum

            thrust = thrust * 200 * thrust_sum_scalar_4


        if self.mode == "Defensive":
            # Defensive Mode
            
            for i, pos in enumerate(relative_positions_sorted):
                asteroid_distance = distances_sorted[i]

                # Skip asteroids that are too far.
                if asteroid_distance > 400:
                    break
            
                # Normalize Distance
                distance_norm = min(50 / (asteroid_distance + EPS), 0.99999)

                # Calculate closure rate.
                closure_rate = _calculate_closure_rate(
                    ship_state["position"],
                    ship_state["heading"],
                    ship_state["speed"],
                    pos,
                    asteroid_velocities_sorted[i],
                )
                closure_rate = min(max((closure_rate+200)/400, 0), 1)

                # Calculate relative heading and normalize.
                relative_heading = _heading_relative_angle([0, 0], ship_state["heading"], pos) / 360
                # Avoid division by zero edge cases.
                if relative_heading == 0 or relative_heading == 1:
                    relative_heading = 0.99999

                # Decide to shoot or run

                defensive_fis_output_1 = _tsk_inference_const(
                    closure_rate,
                    distance_norm,
                    az_mfs_5,
                    distance_mfs_5,
                    rule_constants_thrust_5,
                )

                defensive_fis_output_2 = _tsk_inference_const(
                    relative_heading,
                    defensive_fis_output_1,
                    relative_heading_mfs_6,
                    defensive_fis_mfs_1_6,
                    rule_constants_threat_6,
                )

                # Decide to use avoid mode or shooting mode
                if defensive_fis_output_2 > 0.5:
                    avoid = True
                else:
                    avoid = False
            

            if avoid == True:

                heading_array = []
                for i, pos in enumerate(relative_positions_sorted):

                    relative_heading = _heading_relative_angle([0, 0], ship_state["heading"], pos) / 360
                    heading_array.append(relative_heading)

                aim_point = vm.largest_gap_center(heading_array)
                turn_angle, on_target = vm.go_to_angle(
                ship_state["heading"],
                ship_state["turn_rate_range"],
                aim_point,
                game_state["delta_time"],
                )

                on_target = False

            else:
                threat_index = np.argmax(threat_array)
                
                # Calculate turn angle using the most threatening asteroid.
                turn_angle, on_target = vm.turn_angle(
                    ship_state["position"],
                    ship_state["heading"],
                    ship_state["turn_rate_range"],
                    self.bullet_speed,
                    asteroid_positions_sorted[threat_index],
                    asteroid_velocities_sorted[threat_index],
                    game_state["delta_time"],
                )

                # Determine if we should shoot.
                if on_target:
                    shoot = True


                # === Build Thrust FIS ===


                for i, pos in enumerate(relative_positions_sorted):
                    asteroid_distance = distances_sorted[i]  # Use precomputed distance
                    # Skip asteroids that are too far.
                    if asteroid_distance > 300:
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
                        az_mfs_4,
                        thrust_distance_mfs_4,
                        rule_constants_thrust_4,
                    ) - 0.5

                    thrust += thrust_sum

                thrust = thrust * 200 * thrust_sum_scalar_4


            
        return thrust, turn_angle, shoot, False
