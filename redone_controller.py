from typing import TYPE_CHECKING
from kesslergame import KesslerController
import math
import numpy as np

from utils.kessler_helpers import get_bullet_speed
from TeamTempNameSubmission import vector_math as vm
from algorithms import fuzzy_tree_output, compile_chromosome

if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState

# Total descendant asteroids that spawn when an asteroid of this size is destroyed
_SPAWN_COUNT = {1: 0, 2: 3, 3: 12, 4: 39}

def _asteroid_size(asteroid) -> int:
    """Infer asteroid size tier from its radius."""
    r = asteroid["radius"]
    if r <= 8:
        return 1
    if r <= 16:
        return 2
    if r <= 32:
        return 3
    return 4

def _max_bullets_for(asteroid) -> int:
    """Max bullets we should ever have in flight targeting this asteroid."""
    return min(12, _SPAWN_COUNT.get(_asteroid_size(asteroid)+1, 1))


class FuzzyController(KesslerController):
    def __init__(self, chromosome):
        super().__init__()
        self._name = "BajaBlasteroids"
        self.chromosome = chromosome
        compile_chromosome(self.chromosome)
        self.bullet_speed = get_bullet_speed()

        # Target lock state
        self.locked_target = None
        self.lock_duration = 0

        # FIX 3: Track bullets we've fired that aren't yet in the game's bullet list
        # Each entry: (target_position, target_velocity, frames_remaining)
        self.pending_shots = []

    @property
    def name(self):
        return self._name

    def explanation(self):
        return ""

    def actions(self, ship_state: "ShipOwnState", game_state: "GameState"):
        """Returns (thrust, turn_rate, shoot, deploy_mine)."""
        self.bullet_speed = get_bullet_speed()

        
        # --- Parse ship state ---
        ship_pos        = ship_state["position"]
        ship_heading    = ship_state["heading"]
        ship_speed      = ship_state["speed"]
        ship_vel        = ship_state["velocity"]
        ship_radius     = ship_state["radius"]
        turn_rate_range = ship_state["turn_rate_range"]

        # --- Parse world state ---
        asteroids  = game_state["asteroids"]
        bullets    = game_state["bullets"]
        map_size   = game_state["map_size"]
        delta_time = game_state["delta_time"]

        # FIX 3: Decay pending shot timers each frame
        self.pending_shots = [
            (p, v, f - 1) for p, v, f in self.pending_shots if f > 1
        ]

        if not asteroids:
            self.locked_target = None
            self.lock_duration = 0
            self.pending_shots = []
            return 0.0, 0.0, False, False

        # ---------------------------------
        # FILTER 1: Count bullets already heading for each asteroid
        # ---------------------------------
        bullet_counts = self._count_bullets_per_asteroid(
            asteroids, bullets, self.pending_shots
        )

        # ---------------------------------
        # FILTER 2: Keep asteroids that are hittable AND still need more bullets
        #           - Skip size-1 asteroids (max bullets = 0)
        #           - Skip asteroids already at their bullet cap
        # ---------------------------------
        viable_asteroids = []
        for i, a in enumerate(asteroids):
            max_b = _max_bullets_for(a)
            if max_b <= 0:
                continue  # size-1: don't waste bullets
            if bullet_counts.get(i, 0) >= max_b:
                continue  # already saturated
            # FIX 2: Advance asteroid position by one frame for timing compensation
            adv_pos = (
                a["position"][0] + a["velocity"][0] * delta_time,
                a["position"][1] + a["velocity"][1] * delta_time,
            )
            t_intercept = vm.solve_intercept_time(
                ship_pos, ship_vel, self.bullet_speed, adv_pos, a["velocity"]
            )
            if t_intercept is not None:
                viable_asteroids.append(a)

        # Fallback: if nothing viable, include unsaturated asteroids anyway
        if not viable_asteroids:
            viable_asteroids = [
                a for i, a in enumerate(asteroids)
                if _max_bullets_for(a) > 0
                and bullet_counts.get(i, 0) < _max_bullets_for(a)
            ]

        # Nothing left at all — wait
        if not viable_asteroids:
            self.locked_target = None
            self.lock_duration = 0
            return 0.0, 0.0, False, False

        # ---------------------------------
        # TARGET SELECTION (with lock persistence)
        # ---------------------------------
        target = self._maintain_or_acquire_target(
            viable_asteroids, ship_pos, ship_vel,
            ship_heading, ship_speed, ship_radius, map_size
        )

        # ---------------------------------
        # AIMING  (FIX 1: use direct position — no wrapping)
        # ---------------------------------
        # FIX 2: advance target one frame for timing compensation
        adv_target_pos = (
            target["position"][0] + target["velocity"][0] * delta_time,
            target["position"][1] + target["velocity"][1] * delta_time,
        )

        dist_to_target = math.hypot(
            adv_target_pos[0] - ship_pos[0],
            adv_target_pos[1] - ship_pos[1],
        )
        aim_tolerance = 0.5 if dist_to_target < 200 else 0.0

        turn_rate, on_target = vm.turn_angle(
            ship_pos, ship_vel, ship_heading, turn_rate_range,
            self.bullet_speed, adv_target_pos, target["velocity"],
            delta_time, extra_tolerance=aim_tolerance,
        )

        # Double-check: can we actually hit this target right now?
        if on_target:
            t_check = vm.solve_intercept_time(
                ship_pos, ship_vel, self.bullet_speed,
                adv_target_pos, target["velocity"],
            )
            if t_check is None:
                on_target = False

        # Check if target still has room for more bullets
        pending_count = self._count_pending_for_target(target)
        max_allowed = _max_bullets_for(target)
        can_shoot = on_target and pending_count < max_allowed

        shoot = can_shoot

        # When firing, record the pending shot
        if shoot:
            turn_rate = 0.0
            t_impact = vm.solve_intercept_time(
                ship_pos, ship_vel, self.bullet_speed,
                adv_target_pos, target["velocity"],
            )
            ttl = int((t_impact or 2.0) / max(delta_time, 1e-6)) + 5
            self.pending_shots.append(
                (target["position"], target["velocity"], ttl)
            )
            # If we've now saturated this target, release lock to find next
            if pending_count + 1 >= max_allowed:
                self.locked_target = None
                self.lock_duration = 0
            else:
                # Keep lock — we want to keep firing at this target
                self.locked_target = target
        else:
            self.locked_target = target

        if not math.isfinite(turn_rate):
            turn_rate = 0.0

        # ---------------------------------
        # THROTTLE (collision avoidance + centering)
        # ---------------------------------
        thrust = self._compute_thrust(
            ship_pos, ship_heading, ship_speed, ship_radius, asteroids, map_size
        )

        return thrust, turn_rate, shoot, False

    # =====================================================================
    # PRIVATE HELPERS
    # =====================================================================

    def _count_pending_for_target(self, target):
        """Count how many pending bullets are aimed at this target."""
        t_pos = np.array(target["position"])
        count = 0
        for p, v, f in self.pending_shots:
            if np.linalg.norm(t_pos - np.array(p)) < 80.0:
                count += 1
        return count

    @staticmethod
    def _count_bullets_per_asteroid(asteroids, bullets, pending_shots):
        """Count how many bullets (real + pending) are heading for each asteroid.

        Returns dict of {asteroid_index: bullet_count}.
        """
        counts = {}

        # --- Count real game bullets ---
        for b in bullets:
            b_pos = np.array(b["position"])
            b_vel = np.array(b["velocity"])
            closest_t = float('inf')
            hit_idx = None

            for i, a in enumerate(asteroids):
                a_pos = np.array(a["position"])
                rel_pos = a_pos - b_pos
                rel_vel = np.array(a["velocity"]) - b_vel
                speed_sq = np.dot(rel_vel, rel_vel)
                if speed_sq <= 0:
                    continue
                t_closest = -np.dot(rel_pos, rel_vel) / speed_sq
                if 0 < t_closest < 3.0 and t_closest < closest_t:
                    dist_at_t = np.linalg.norm(rel_pos + rel_vel * t_closest)
                    if dist_at_t < a["radius"]:
                        closest_t = t_closest
                        hit_idx = i

            if hit_idx is not None:
                counts[hit_idx] = counts.get(hit_idx, 0) + 1

        # --- Count pending shots ---
        for p, v, f in pending_shots:
            p_arr = np.array(p)
            closest_dist = float('inf')
            closest_idx = None
            for i, a in enumerate(asteroids):
                dist = np.linalg.norm(np.array(a["position"]) - p_arr)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            if closest_idx is not None and closest_dist < 80.0:
                counts[closest_idx] = counts.get(closest_idx, 0) + 1

        return counts

    def _maintain_or_acquire_target(self, viable_asteroids, ship_pos, ship_vel,
                                     ship_heading, ship_speed, ship_radius, map_size):
        """Try to keep current lock; if lost or timed-out, pick a new target via fuzzy eval."""
        target = None

        # --- Try to maintain existing lock ---
        if self.locked_target is not None:
            last_pos = np.array(self.locked_target["position"])
            closest_dist = float('inf')
            closest_match = None

            for a in viable_asteroids:
                dist = np.linalg.norm(np.array(a["position"]) - last_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_match = a

            if closest_match is not None and closest_dist < 80.0:
                target = closest_match
                self.lock_duration += 1
            else:
                self.locked_target = None
                self.lock_duration = 0

            # Timeout: if we've been locked >20 frames without shooting, re-evaluate
            if self.lock_duration > 20:
                self.locked_target = None
                self.lock_duration = 0
                target = None

        # --- Fuzzy-tree target selection ---
        if target is None:
            target = self._fuzzy_select_target(
                viable_asteroids, ship_pos, ship_vel,
                ship_heading, ship_speed, ship_radius, map_size,
            )
            self.locked_target = target
            self.lock_duration = 0

        return target

    def _fuzzy_select_target(self, viable_asteroids, ship_pos, ship_vel,
                              ship_heading, ship_speed, ship_radius, map_size):
        """Score each viable asteroid with the evolved fuzzy tree; return the best.

        NOTE: Target PRIORITIZATION still uses wrapped distances (the ship itself
        wraps, so a nearby-via-wrap asteroid is genuinely close for selection).
        Actual aiming/shooting uses direct positions — handled by the caller.
        """
        n = len(viable_asteroids)
        inputs_batch = np.zeros((n, 5), dtype=np.float64)
        max_dist = math.sqrt(map_size[0]**2 + map_size[1]**2)
        log_max_dist = math.log(max_dist + 1)
        ship_vel_np = np.array(ship_vel)

        for i, asteroid in enumerate(viable_asteroids):
            apos = asteroid["position"]
            avel = asteroid["velocity"]
            arad = asteroid["radius"]

            # Use wrapped delta for SELECTION scoring (ship wraps)
            w_dx, w_dy = vm.wrapped_delta(ship_pos, apos, map_size)
            virtual_pos = (ship_pos[0] + w_dx, ship_pos[1] + w_dy)

            # Input 0: Heading alignment (1 = ahead, 0 = behind)
            rel_ang = vm.heading_relative_angle(ship_pos, ship_heading, virtual_pos)
            input_heading = 2.0 * abs((rel_ang / 360.0) - 0.5)

            # Input 1: Closure rate (0..1)
            closure = vm.calculate_closure_rate(
                ship_pos, ship_heading, ship_speed, virtual_pos, avel
            )
            input_closure = np.clip((closure + 100) / 200, 0.0, 1.0)

            # Input 2: Asteroid radius (0..1)
            input_radius = np.clip(arad / 40.0, 0.0, 1.0)
            
            # Input 3: Log distance (0..1)
            dist_val = math.hypot(w_dx, w_dy)
            input_distance = np.clip(math.log(dist_val + 1) / log_max_dist, 0.0, 1.0)

            # Input 4: Collision urgency (0..1)
            rel_pos = np.array([w_dx, w_dy])
            rel_vel = np.array(avel) - ship_vel_np
            speed_sq = np.dot(rel_vel, rel_vel)
            input_collision = 0.0
            if speed_sq > 0:
                t_closest = -np.dot(rel_pos, rel_vel) / speed_sq
                if t_closest > 0:
                    dist_at_t = np.linalg.norm(rel_pos + rel_vel * t_closest)
                    if dist_at_t < (ship_radius + arad + 5.0):
                        input_collision = np.clip(
                            (5.0 - t_closest) / 5.0, 0.0, 1.0
                        )

            inputs_batch[i] = [
                input_heading, input_closure, input_radius,
                input_distance, input_collision,
            ]

        scores = fuzzy_tree_output(self.chromosome, inputs_batch)
        return viable_asteroids[int(np.argmax(scores))]

    @staticmethod
    def _compute_thrust(ship_pos, ship_heading, ship_speed, ship_radius,
                         asteroids, map_size):
        """Simple collision-avoidance thrust + gentle centering."""
        asteroid_positions = [a["position"] for a in asteroids]
        relative_positions = vm.game_to_ship_frame(
            ship_pos, asteroid_positions, map_size
        )

        closest_dist = float('inf')
        closest_rel = None
        for rel_pos in relative_positions:
            dist = math.hypot(rel_pos[0], rel_pos[1])
            if dist < closest_dist:
                closest_dist = dist
                closest_rel = np.array(rel_pos)

        rad = math.radians(ship_heading)
        ship_dir = np.array([math.cos(rad), math.sin(rad)])

        # Collision avoidance
        if closest_rel is not None and closest_dist < 250.0:
            return -480.0 if np.dot(closest_rel, ship_dir) > 0 else 480.0

        # Gentle centering / drag
        center = np.array([map_size[0] / 2, map_size[1] / 2])
        to_center = center - np.array(ship_pos)
        dist_to_center = np.linalg.norm(to_center)
        edge_threshold = min(map_size[0], map_size[1]) * 0.25

        if dist_to_center > edge_threshold:
            to_center_norm = to_center / max(dist_to_center, 1.0)
            return float(np.dot(to_center_norm, ship_dir) * 120.0)

        if ship_speed > 10.0:
            return -200.0

        return 0.0