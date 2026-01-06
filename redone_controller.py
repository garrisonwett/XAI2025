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
        # NEW: Counter for how long we've held a lock
        self.lock_duration = 0

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
            self.lock_duration = 0 # NEW: Reset duration
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
             self.lock_duration = 0 # NEW: Reset duration
             return 0.0, 0.0, False, False

        # ---------------------------------
        # TARGET LOCKING LOGIC
        # ---------------------------------
        target = None

        # NEW: Check if lock duration has exceeded 4 timesteps
        # If so, force a drop before we even try to maintain it.
        if self.locked_target is not None and self.lock_duration >= 4:
            self.locked_target = None
            self.lock_duration = 0
        
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
                # NEW: Increment duration since we maintained the lock
                self.lock_duration += 1
            else:
                self.locked_target = None
                self.lock_duration = 0 # NEW: Reset on lost lock

        # 2. If we don't have a lock (or just dropped it due to expiry/loss), calculate new target
        if target is None:
            # NEW: Reset duration because we are picking a brand new target
            self.lock_duration = 0

            # BATCH INPUT CALCULATION (NOW 5 INPUTS)
            num_asteroids = len(viable_asteroids)
            inputs_batch = np.zeros((num_asteroids, 5), dtype=np.float64) 
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
                input_radius = np.clip(arad/40, 0.0, 1.0)
                
                # 4. Distance
                dist_val = math.hypot(apos[0] - ship_pos[0], apos[1] - ship_pos[1])
                input_distance = np.clip(dist_val / max_dist, 0.0, 1.0)

                # 5. Collision Threat
                rel_pos = np.array(apos) - np.array(ship_pos)
                rel_vel = np.array(avel) - ship_vel_np
                
                v_dot_v = np.dot(rel_vel, rel_vel)
                
                input_collision = 0.0
                
                if v_dot_v < 0: # Negative dot product means closing distance
                    
                    speed_sq = np.dot(rel_vel, rel_vel)
                    if speed_sq > 0:
                        t_closest = -np.dot(rel_pos, rel_vel) / speed_sq
                        
                        if t_closest > 0:
                            dist_at_t = np.linalg.norm(rel_pos + rel_vel * t_closest)
                            
                            if dist_at_t < (ship_radius + arad + 5.0):
                                 input_collision = np.clip((10.0 - t_closest) / 10.0, 0.0, 1.0)

                inputs_batch[i, 0] = input_heading
                inputs_batch[i, 1] = input_closure
                inputs_batch[i, 2] = input_radius
                inputs_batch[i, 3] = input_distance
                inputs_batch[i, 4] = input_collision 

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

        # 1. Calculate Proximity (Existing logic reused)
        closest_dist = float('inf')
        closest_rel_pos = None # Vector pointing FROM ship TO asteroid
        
        asteroid_positions = [a["position"] for a in asteroids]
        # vm.game_to_ship_frame handles the map wrapping math for us
        relative_positions = vm.game_to_ship_frame(ship_pos, asteroid_positions, game_state["map_size"])

        for rel_pos in relative_positions:
            dist = math.hypot(rel_pos[0], rel_pos[1])
            if dist < closest_dist:
                closest_dist = dist
                closest_rel_pos = np.array(rel_pos)

        # 2. Determine Thrust
        thrust = 0.0
        
        # Calculate ship direction vector
        rad = math.radians(ship_heading)
        ship_dir = np.array([math.cos(rad), math.sin(rad)])

        # A. CRITICAL EVASION (Panic Zone)
        # If an asteroid is within 250 units, get away immediately.
        if closest_rel_pos is not None and closest_dist < 250.0:
            # Dot product determines if asteroid is generally in front (>0) or behind (<0)
            if np.dot(closest_rel_pos, ship_dir) > 0:
                thrust = -480.0  # Asteroid is in front -> Full Reverse
            else:
                thrust = 480.0   # Asteroid is behind -> Full Forward
        
        # B. STABILITY (Braking Zone)
        # If safe, apply drag to stop drifting. This makes aiming much easier.
        elif ship_speed > 10.0:
             thrust = -200.0  # Apply gentle reverse thrust to slow down
        
        # (Note: We removed the 'thrust = 0' line that was overwriting your logic)

        if not math.isfinite(turn_angle):
            turn_angle = 0.0
            
        turn_rate = turn_angle
        shoot = on_target
        deploy_mine = False

        # Unlock after shooting so we can re-evaluate targets
        if shoot:
            self.locked_target = None
            self.lock_duration = 0 # NEW: Reset duration after shooting

        return thrust, turn_rate, shoot, deploy_mine