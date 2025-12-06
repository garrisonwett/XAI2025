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
        
        # State variable for target locking
        self.locked_target = None

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
        ship_vel          = ship_state["velocity"]
        ship_radius       = ship_state["radius"]
        
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
            self.locked_target = None
            return 0.0, 0.0, False, False

        # ---------------------------------
        # FILTER 1: IGNORE DOOMED ASTEROIDS
        # ---------------------------------
        doomed_indices = set()
        
        for b in bullets:
            b_pos = np.array(b["position"])
            b_vel = np.array(b["velocity"])

            for i, a in enumerate(asteroids):
                if i in doomed_indices:
                    continue

                a_pos = np.array(a["position"])
                a_vel = np.array(a["velocity"])
                a_rad = a["radius"]

                rel_pos = a_pos - b_pos
                rel_vel = a_vel - b_vel
                
                v_dot_v = np.dot(rel_vel, rel_vel)
                
                if v_dot_v > 0:
                    t_closest = -np.dot(rel_pos, rel_vel) / v_dot_v
                    if t_closest > 0 and t_closest < 3.0:
                        dist_at_t = np.linalg.norm(rel_pos + rel_vel * t_closest)
                        if dist_at_t < (a_rad):
                            doomed_indices.add(i)
                            break 

        # ---------------------------------
        # FILTER 2: IGNORE WRAPPING ASTEROIDS
        # ---------------------------------
        wrapping_indices = set()

        for i, a in enumerate(asteroids):
            if i in doomed_indices: continue

            a_pos = np.array(a["position"])
            a_vel = np.array(a["velocity"])
            
            # Distance to ship
            dist = np.linalg.norm(a_pos - np.array(ship_pos))
            if dist < 1.0: dist = 1.0

            t_impact = dist / self.bullet_speed
            
            # Estimate turn time (worst case 90 degree turn)
            t_turn = 90.0 / max(abs(turn_rate_range[0]), abs(turn_rate_range[1]), 1.0)
            total_time = t_impact + t_turn

            future_pos = a_pos + a_vel * total_time

            if (future_pos[0] < 0 or future_pos[0] > map_width or
                future_pos[1] < 0 or future_pos[1] > map_height):
                wrapping_indices.add(i)

        # ---------------------------------
        # COMPILE VIABLE LIST
        # ---------------------------------
        viable_asteroids = []
        for i, a in enumerate(asteroids):
            if i not in doomed_indices and i not in wrapping_indices:
                viable_asteroids.append(a)

        # Fallback if filters removed everything
        if not viable_asteroids:
             viable_asteroids = [a for i, a in enumerate(asteroids) if i not in doomed_indices]
        
        if not viable_asteroids:
             self.locked_target = None
             return 0.0, 0.0, False, False

        # ---------------------------------
        # TARGET LOCKING LOGIC
        # ---------------------------------
        target = None
        
        # 1. Try to maintain the existing lock
        if self.locked_target is not None:
            closest_dist = float('inf')
            closest_match = None
            last_pos = np.array(self.locked_target["position"])
            
            for a in viable_asteroids:
                curr_pos = np.array(a["position"])
                dist = np.linalg.norm(curr_pos - last_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_match = a
            
            if closest_match is not None and closest_dist < 100.0:
                target = closest_match
            else:
                self.locked_target = None

        # 2. If we don't have a lock, calculate new target
        if target is None:
            # BATCH INPUT CALCULATION (NOW 5 INPUTS)
            num_asteroids = len(viable_asteroids)
            inputs_batch = np.zeros((num_asteroids, 5), dtype=np.float64) # <--- SIZE 5
            max_dist = math.sqrt(map_width**2 + map_height**2)

            ship_vel_np = np.array(ship_vel)

            for i, asteroid in enumerate(viable_asteroids):
                apos = asteroid["position"]
                avel = asteroid["velocity"]
                arad = asteroid["radius"]

                # 1. Relative Heading
                rel_ang_deg = vm.heading_relative_angle(ship_pos, ship_heading, apos)
                rel_heading_norm = rel_ang_deg / 360.0
                input_heading = abs(rel_heading_norm - 1.0)

                # 2. Closure Rate
                closure = vm.calculate_closure_rate(ship_pos, ship_heading, ship_speed, apos, avel)
                input_closure = np.clip((closure + 100) / 200, 0.0, 1.0)

                # 3. Radius
                input_radius = np.clip((arad - 8) / 24, 0.0, 1.0)
                
                # 4. Distance
                dist_val = math.hypot(apos[0] - ship_pos[0], apos[1] - ship_pos[1])
                input_distance = np.clip(dist_val / max_dist, 0.0, 1.0)

                # 5. Collision Threat (NEW INPUT)
                # Calculates time until asteroid hits ship
                rel_pos = np.array(apos) - np.array(ship_pos)
                rel_vel = np.array(avel) - ship_vel_np
                
                v_dot_v = np.dot(rel_vel, rel_vel)
                
                input_collision = 0.0
                
                if v_dot_v < 0: # Negative dot product means closing distance
                    # Time to closest approach
                    # t = - (r . v) / (v . v)
                    # Note: Since we check v_dot_v < 0 (closing), we actually want to divide by speed^2 (norm)
                    # which is v_dot_v. But dot product of closing vectors is negative?
                    # Let's check math: 
                    # Closest approach time t = -(P . V) / |V|^2
                    # If P points to Asteroid, V is Ast_Vel relative to Ship.
                    # If V is towards Ship, P . V is negative. -(neg) is positive. Correct.
                    
                    speed_sq = np.dot(rel_vel, rel_vel)
                    if speed_sq > 0:
                        t_closest = -np.dot(rel_pos, rel_vel) / speed_sq
                        
                        if t_closest > 0:
                            dist_at_t = np.linalg.norm(rel_pos + rel_vel * t_closest)
                            
                            # If it passes within collision radius + small buffer
                            if dist_at_t < (ship_radius + arad + 5.0):
                                 # Normalize Time: 0s = 1.0 (Danger), 10s = 0.0 (Safe)
                                 # 10 seconds is a reasonable "panic" horizon
                                 input_collision = np.clip((10.0 - t_closest) / 10.0, 0.0, 1.0)

                inputs_batch[i, 0] = input_heading
                inputs_batch[i, 1] = input_closure
                inputs_batch[i, 2] = input_radius
                inputs_batch[i, 3] = input_distance
                inputs_batch[i, 4] = input_collision  # <--- ASSIGN NEW INPUT

            # Batch Fuzzy Eval
            threat_scores = fuzzy_tree_output(self.chromosome, inputs_batch)
            best_idx = np.argmax(threat_scores)
            
            target = viable_asteroids[best_idx]
            
            # ACQUIRE LOCK
            self.locked_target = target

        # Ensure lock is updated to current frame object
        self.locked_target = target

        # ---------------------------------
        # AIMING & ACTIONS
        # ---------------------------------
        target_pos = target["position"]
        target_vel = target["velocity"]

        # 0.0 = Precise Hit, 0.5 = Loose Aim
        aim_tolerance = 0.5 

        turn_angle, on_target = vm.turn_angle(
            ship_pos,
            ship_heading,
            turn_rate_range,
            self.bullet_speed,
            target_pos,
            target_vel,
            delta_time,
            extra_tolerance=aim_tolerance
        )


        
        # ###############
        # # THROTTLE LOGIC
        thrust = 0.0

        # Simple Proximity Avoidance (Accounting for toroidal map wrapping using copysign)
        closest_dist = float('inf')
        closest_rel_pos = None # Vector pointing to the asteroid relative to ship

        for a in viable_asteroids:
            a_pos = a["position"]
            dx = a_pos[0] - ship_pos[0]
            dy = a_pos[1] - ship_pos[1]

            # Warp helper logic: shortest path across toroidal boundary
            if abs(dx) > map_width / 2:
                dx -= math.copysign(map_width, dx)
            if abs(dy) > map_height / 2:
                dy -= math.copysign(map_height, dy)

            d = math.hypot(dx, dy)

            if d < closest_dist:
                closest_dist = d
                closest_rel_pos = np.array([dx, dy])

        # If the closest asteroid is within 200 units, take evasive action
        if closest_rel_pos is not None and closest_dist < 200.0:
            # Calculate ship direction vector from heading
            rad = math.radians(ship_heading)
            ship_dir = np.array([math.cos(rad), math.sin(rad)])
            
            # Check if asteroid is in front (dot product > 0) or behind
            # closest_rel_pos is the vector pointing FROM ship TO asteroid
            if np.dot(closest_rel_pos, ship_dir) > 0:
                thrust = -480.0 * min(1, 50/closest_dist)  # Reverse away from danger
            else:
                thrust = 480.0 * min(1, 50/closest_dist)  # Accelerate away from danger







        if not math.isfinite(turn_angle):
            turn_angle = 0.0
            
        turn_rate = turn_angle
        shoot = on_target
        deploy_mine = False

        # Unlock after shooting so we can re-evaluate targets
        if shoot:
            self.locked_target = None

        return thrust, turn_rate, shoot, deploy_mine
    