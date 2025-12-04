from typing import TYPE_CHECKING
from kesslergame import KesslerController
import math
import numpy as np

from utils.kessler_helpers import get_bullet_speed
from TeamTempNameSubmission import vector_math as vm
from algorithms import fuzzy_tree_output, compile_chromosome

if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState


class FuzzyController(KesslerController):
    def __init__(self, chromosome):
        super().__init__()
        self._name = "BajaBlasteroids"
        self.chromosome = chromosome

        # Ensure chromosome is compiled for fast execution
        compile_chromosome(self.chromosome)
        self.bullet_speed = get_bullet_speed()

    @property
    def name(self):
        return self._name

    def explanation(self):
        return ""

    def actions(self, ship_state: "ShipOwnState", game_state: "GameState"):
        """
        Returns (thrust, turn_angle, shoot, mine).
        """
        self.bullet_speed = get_bullet_speed()

        # -------------------------------
        # Parse own ship information
        # -------------------------------
        ship_pos          = ship_state["position"]
        ship_heading      = ship_state["heading"]
        ship_speed        = ship_state["speed"]
        
        turn_rate_range   = ship_state["turn_rate_range"]

        # -------------------------------
        # Parse world information
        # -------------------------------
        asteroids   = game_state["asteroids"]
        bullets     = game_state["bullets"]

        map_width, map_height = game_state["map_size"]
        delta_time            = game_state["delta_time"]

        # If no asteroids, sit still and don't shoot
        if not asteroids:
            return 0.0, 0.0, False, False

        # ---------------------------------
        # FILTER: IGNORE DOOMED ASTEROIDS
        # ---------------------------------
        # We track asteroids by INDEX (0, 1, 2...) because 'id' might be missing
        doomed_indices = set()
        
        for b in bullets:
            b_pos = np.array(b["position"])
            b_vel = np.array(b["velocity"])

            for i, a in enumerate(asteroids):
                # Skip if we already know this asteroid is dead
                if i in doomed_indices:
                    continue

                a_pos = np.array(a["position"])
                a_vel = np.array(a["velocity"])
                a_rad = a["radius"]

                # Vector Math to predict collision
                rel_pos = a_pos - b_pos
                rel_vel = a_vel - b_vel
                
                v_dot_v = np.dot(rel_vel, rel_vel)
                
                # Check if moving towards each other
                if v_dot_v > 0:
                    t_closest = -np.dot(rel_pos, rel_vel) / v_dot_v
                    
                    # Check if collision is in the near future (0 to 3 seconds)
                    if t_closest > 0 and t_closest < 3.0:
                        dist_at_t = np.linalg.norm(rel_pos + rel_vel * t_closest)
                        
                        # Check collision radius
                        if dist_at_t < (a_rad):
                            doomed_indices.add(i)
                            break # Bullet used up on this asteroid

        # Create a list of asteroids that are NOT doomed
        viable_asteroids = [a for i, a in enumerate(asteroids) if i not in doomed_indices]

        # If all asteroids are doomed, sit tight
        if not viable_asteroids:
             return 0.0, 0.0, False, False

        # ---------------------------------
        # BATCH INPUT CALCULATION
        # ---------------------------------
        num_asteroids = len(viable_asteroids)
        inputs_batch = np.zeros((num_asteroids, 4), dtype=np.float64)
        
        max_dist = math.sqrt(map_width**2 + map_height**2)

        for i, asteroid in enumerate(viable_asteroids):
            apos = asteroid["position"]
            avel = asteroid["velocity"]
            arad = asteroid["radius"]

            # 1. Relative Heading
            rel_ang_deg = vm.heading_relative_angle(ship_pos, ship_heading, apos)
            rel_heading_norm = rel_ang_deg / 360.0
            input_heading = abs(rel_heading_norm - 1.0)

            # 2. Closure Rate
            closure = vm.calculate_closure_rate(
                ship_pos, ship_heading, ship_speed, apos, avel
            )
            input_closure = np.clip((closure + 100) / 200, 0.0, 1.0)

            # 3. Radius
            input_radius = np.clip((arad - 8) / 24, 0.0, 1.0)

            # 4. Distance
            dist_val = math.hypot(apos[0] - ship_pos[0], apos[1] - ship_pos[1])
            input_distance = np.clip(dist_val / max_dist, 0.0, 1.0)

            inputs_batch[i, 0] = input_heading
            inputs_batch[i, 1] = input_closure
            inputs_batch[i, 2] = input_radius
            inputs_batch[i, 3] = input_distance

        # ---------------------------------
        # BATCH FUZZY EVALUATION
        # ---------------------------------
        threat_scores = fuzzy_tree_output(self.chromosome, inputs_batch)

        # ---------------------------------
        # SELECTION & TARGETING
        # ---------------------------------
        best_idx = np.argmax(threat_scores)
        target = viable_asteroids[best_idx]
        
        target_pos = target["position"]
        target_vel = target["velocity"]

        turn_angle, on_target = vm.turn_angle(
            ship_pos,
            ship_heading,
            turn_rate_range,
            self.bullet_speed,
            target_pos,
            target_vel,
            delta_time
        )

        # ---------------------------------
        # ACTIONS
        # ---------------------------------
        thrust = 0.0
        
        # SAFETY CHECK: Ensure turn_rate is finite
        if not math.isfinite(turn_angle):
            turn_angle = 0.0
            
        turn_rate = turn_angle
        shoot = on_target
        deploy_mine = False

        return thrust, turn_rate, shoot, deploy_mine