from typing import TYPE_CHECKING
from kesslergame import KesslerController

from utils import LoggerUtility
from utils.kessler_helpers import get_bullet_speed
from TeamTempNameSubmission import vector_math as vm
from TeamTempNameSubmission import fuzzy_trees as ft

if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState

import math
import numpy as np


class FuzzyController(KesslerController):
    """
    GA driven fuzzy controller.

    All tunable quantities in these categories are derived from a chromosome
    of length 177 with values in [0,1]:

      1. Scaling values (8 genes)
         - Distance, closure, thrust, turn, mode hysteresis thresholds

      2. Membership function centers (32 genes)
         - Centers for every input MF set in all FIS

      3. TSK rule consequent parameters (132 genes)
         - Each rule has [p0, p1, p2] taken from the chromosome

    Logic wiring, which inputs feed which FIS, and logging remain fixed.

    Chromosome length: 177
    """

    def __init__(self):
        super().__init__()
        self._name = "BajaBlasteroids_Fuzzy_GA"

        # Persistent asteroid ID machinery
        self._tracked_asteroids: dict[int, tuple[float, float]] = {}
        self._next_asteroid_id = 0

        # Shot history
        self.asteroids_shot_at: list[int] = []

        # Respawn timer
        self.respawn_time = 0.0

        # Frame counter
        self.frame_index = 0

        # In memory log of all frames
        self.data_log: list[dict] = []

        # Optional logger utility
        self.logger = LoggerUtility()

        # Current mode and hysteresis support
        self.mode = "Defensive"
        self._mode_safety_score = 0.0

        # Scaling values (set from chromosome in _build_fuzzy_from_chromosome)
        self.max_safe_distance = 600.0
        self.max_threat_distance = 600.0
        self.max_closure_mag = 200.0
        self.off_thrust_gain = 1.0
        self.def_thrust_gain = 1.0
        self.turn_gain = 1.0
        self.mode_enter_threshold = 0.6
        self.mode_exit_threshold = 0.45

        # Defensive shooting thresholds (from chromosome)
        self.shoot_dist_threshold = 150.0
        self.shoot_threat_threshold = 0.7

        # Respawn timing (from chromosome)
        self.respawn_total_time = 3.0
        self.respawn_stage1_time = 2.0
        self.respawn_stage2_time = 1.0

        # Bullet speed for aiming
        self.bullet_speed = get_bullet_speed()

        # Fuzzy system containers (MFs and params)
        self._init_fuzzy_placeholders()

    # ------------------------------------------------------------------
    # Fuzzy placeholders
    # ------------------------------------------------------------------

    def _init_fuzzy_placeholders(self) -> None:
        # Membership functions for each FIS will be set from chromosome
        self.mode_safe_dist_mfs = []
        self.mode_avg_threat_mfs = []
        self.threat_dist_mfs = []
        self.threat_closure_mfs = []
        self.off_thrust_dist_mfs = []
        self.off_thrust_threat_mfs = []
        self.def_urg_safe_dist_mfs = []
        self.def_urg_max_threat_mfs = []
        self.def_steer_ang_err_mfs = []
        self.def_steer_urg_mfs = []
        self.def_thrust_urg_mfs = []
        self.def_thrust_safe_dist_mfs = []

        # TSK params for each FIS will be set from chromosome
        self.mode_params = None
        self.threat_params = None
        self.off_thrust_params = None
        self.def_urg_params = None
        self.def_steer_params = None
        self.def_thrust_params = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def explanation(self) -> str:
        return f"Mode: {self.mode}, safety_score={self._mode_safety_score:.3f}"

    # ------------------------------------------------------------------
    # Reset and logging
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        self._tracked_asteroids.clear()
        self._next_asteroid_id = 0
        self.asteroids_shot_at.clear()
        self.respawn_time = 0.0
        self.frame_index = 0
        self.data_log.clear()
        self.mode = "Defensive"
        self._mode_safety_score = 0.0

    def _assign_persistent_ids(
        self,
        asteroids: list[dict],
        dt: float,
    ) -> tuple[
        dict[int, tuple[float, float]],
        list[int],
        list[tuple[float, float]],
        list[tuple[float, float]],
    ]:
        """
        Match current asteroids to previous ones using predicted last positions.
        """
        _hypot = math.hypot

        world_positions = [tuple(a["position"]) for a in asteroids]
        velocities = [tuple(a["velocity"]) for a in asteroids]

        new_tracked: dict[int, tuple[float, float]] = {}
        used_old_ids = set()

        for wpos, vel in zip(world_positions, velocities):
            pred_x = wpos[0] - vel[0] * dt
            pred_y = wpos[1] - vel[1] * dt
            best_id = None
            best_dist = float("inf")

            for aid, last_pos in self._tracked_asteroids.items():
                if aid in used_old_ids:
                    continue
                d = _hypot(last_pos[0] - pred_x, last_pos[1] - pred_y)
                if d < best_dist:
                    best_dist = d
                    best_id = aid

            vel_norm = _hypot(vel[0], vel[1])
            thresh = vel_norm * dt * 1.5 + 1e-3

            if best_id is not None and best_dist <= thresh:
                aid = best_id
            else:
                aid = self._next_asteroid_id
                self._next_asteroid_id += 1

            new_tracked[aid] = wpos
            used_old_ids.add(aid)

        self._tracked_asteroids = new_tracked
        ids_sorted = list(new_tracked.keys())
        return new_tracked, ids_sorted, world_positions, velocities

    def _log_frame(
        self,
        ship_state: "ShipOwnState",
        game_state: "GameState",
        asteroids: list[dict],
        tracked_asteroids: dict[int, tuple[float, float]],
        world_positions: list[tuple[float, float]],
        velocities: list[tuple[float, float]],
        mode: str,
        safety_score: float,
    ) -> None:
        _hypot = math.hypot

        rel_positions = vm.game_to_ship_frame(
            ship_state["position"],
            world_positions,
            game_state["map_size"],
        )

        asteroid_records = []
        for aid, rpos, vel, wpos, raw in zip(
            tracked_asteroids.keys(),
            rel_positions,
            velocities,
            world_positions,
            asteroids,
        ):
            dist = _hypot(rpos[0], rpos[1])

            asteroid_records.append(
                {
                    "id": aid,
                    "world_position": {"x": wpos[0], "y": wpos[1]},
                    "relative_position": {"x": rpos[0], "y": rpos[1]},
                    "velocity": {"vx": vel[0], "vy": vel[1]},
                    "distance": dist,
                    "radius": raw.get("radius"),
                    "mass": raw.get("mass"),
                }
            )

        asteroid_records.sort(key=lambda a: a["distance"])

        ship_record = {
            "position": {
                "x": ship_state["position"][0],
                "y": ship_state["position"][1],
            },
            "velocity": {
                "vx": ship_state.get("velocity", (0.0, 0.0))[0],
                "vy": ship_state.get("velocity", (0.0, 0.0))[1],
            },
            "heading": ship_state["heading"],
            "speed": ship_state["speed"],
            "turn_rate_range": ship_state["turn_rate_range"],
            "can_fire": ship_state["can_fire"],
            "is_respawning": ship_state["is_respawning"],
        }

        meta_record = {
            "frame_index": self.frame_index,
            "time": game_state["time"],
            "delta_time": game_state["delta_time"],
            "map_size": game_state["map_size"],
            "num_asteroids": len(asteroid_records),
            "respawn_time_internal": self.respawn_time,
            "mode": mode,
            "safety_score": safety_score,
        }

        frame_record = {
            "ship": ship_record,
            "asteroids": asteroid_records,
            "meta": meta_record,
        }

        self.data_log.append(frame_record)

        try:
            if hasattr(self.logger, "log"):
                self.logger.log(frame_record)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Chromosome driven fuzzy system builder
    # ------------------------------------------------------------------

    def _build_fuzzy_from_chromosome(self, chromosome_raw) -> None:
        """
        Use values in [0, 1] from chromosome to set:

          1) Scaling values
          2) Membership function centers
          3) TSK rule consequent parameters
          4) Defensive shooting thresholds
          5) Respawn timing thresholds

        Chromosome layout (177 genes total):

          0  7   : scaling values (8)
          8  39  : MF centers (32)
          40 171 : TSK params for all FIS (132)
          172 176: behavior thresholds (5)
        """

        if chromosome_raw is None:
            chrom = np.zeros(177, dtype=float)
        else:
            chrom = np.asarray(chromosome_raw, dtype=float).flatten()
            if chrom.size < 177:
                # pad with mid values if too short
                pad = np.full(177 - chrom.size, 0.5, dtype=float)
                chrom = np.concatenate([chrom, pad])
            elif chrom.size > 177:
                chrom = chrom[:177]

        idx = 0

        def grab(count, default_val=0.5):
            nonlocal idx
            if count <= 0:
                return []
            vals = chrom[idx : idx + count]
            if vals.size < count:
                extra = np.full(count - vals.size, default_val, dtype=float)
                vals = np.concatenate([vals, extra])
            idx += count
            # clamp to [0,1]
            vals = np.clip(vals, 0.0, 1.0)
            return vals.tolist()

        # 1) Scaling values (8 genes)

        # distance scales: map [0,1] to [100, 1500]
        safe_scale_raw, threat_scale_raw = grab(2)
        self.max_safe_distance = 100.0 + safe_scale_raw * 1400.0
        self.max_threat_distance = 100.0 + threat_scale_raw * 1400.0

        # closure scale: map [0,1] to [50, 400]
        closure_scale_raw = grab(1)[0]
        self.max_closure_mag = 50.0 + closure_scale_raw * 350.0

        # thrust gains: map [0,1] to [0.5, 2.0]
        off_thrust_gain_raw, def_thrust_gain_raw = grab(2)
        self.off_thrust_gain = 0.5 + off_thrust_gain_raw * 1.5
        self.def_thrust_gain = 0.5 + def_thrust_gain_raw * 1.5

        # turn gain: map [0,1] to [0.5, 2.0]
        turn_gain_raw = grab(1)[0]
        self.turn_gain = 0.5 + turn_gain_raw * 1.5

        # mode hysteresis thresholds using center and width
        mode_center_raw, mode_width_raw = grab(2)
        width = 0.05 + 0.45 * mode_width_raw   # 0.05 to 0.5
        center = max(0.0, min(mode_center_raw, 1.0))
        low = max(0.0, center - width / 2.0)
        high = min(1.0, center + width / 2.0)
        self.mode_exit_threshold = low
        self.mode_enter_threshold = high

        # 2) Membership function centers (32 genes)

        def sorted_centers(vals):
            return sorted(max(0.0, min(v, 1.0)) for v in vals)

        # mode selection FIS
        mode_safe_centers = sorted_centers(grab(3))
        mode_threat_centers = sorted_centers(grab(3))
        self.mode_safe_dist_mfs = ft.build_triangles(mode_safe_centers)
        self.mode_avg_threat_mfs = ft.build_triangles(mode_threat_centers)

        # threat FIS
        threat_dist_centers = sorted_centers(grab(3))
        threat_closure_centers = sorted_centers(grab(3))
        self.threat_dist_mfs = ft.build_triangles(threat_dist_centers)
        self.threat_closure_mfs = ft.build_triangles(threat_closure_centers)

        # offensive thrust FIS
        off_dist_centers = sorted_centers(grab(2))
        off_threat_centers = sorted_centers(grab(2))
        self.off_thrust_dist_mfs = ft.build_triangles(off_dist_centers)
        self.off_thrust_threat_mfs = ft.build_triangles(off_threat_centers)

        # defensive urgency FIS
        def_urg_safe_centers = sorted_centers(grab(3))
        def_urg_max_centers = sorted_centers(grab(3))
        self.def_urg_safe_dist_mfs = ft.build_triangles(def_urg_safe_centers)
        self.def_urg_max_threat_mfs = ft.build_triangles(def_urg_max_centers)

        # defensive steering FIS
        def_steer_ang_centers = sorted_centers(grab(3))
        def_steer_urg_centers = sorted_centers(grab(3))
        self.def_steer_ang_err_mfs = ft.build_triangles(def_steer_ang_centers)
        self.def_steer_urg_mfs = ft.build_triangles(def_steer_urg_centers)

        # defensive thrust FIS
        def_thrust_urg_centers = sorted_centers(grab(2))
        def_thrust_safe_centers = sorted_centers(grab(2))
        self.def_thrust_urg_mfs = ft.build_triangles(def_thrust_urg_centers)
        self.def_thrust_safe_dist_mfs = ft.build_triangles(def_thrust_safe_centers)

        # 3) TSK rule consequent parameters (132 genes)

        def build_tsk_params(mfs1, mfs2):
            n1 = len(mfs1)
            n2 = len(mfs2)
            count = n1 * n2 * 3
            raw = grab(count)
            params = []
            k = 0
            for i in range(n1):
                row = []
                for j in range(n2):
                    r0 = raw[k]
                    r1 = raw[k + 1]
                    r2 = raw[k + 2]
                    k += 3
                    # Map raw in [0,1] to meaningful ranges
                    # p0 in [-0.5, 0.5], p1, p2 in [-1, 1]
                    p0 = (r0 - 0.5) * 1.0
                    p1 = (r1 * 2.0) - 1.0
                    p2 = (r2 * 2.0) - 1.0
                    row.append([p0, p1, p2])
                params.append(row)
            return params

        self.mode_params = build_tsk_params(self.mode_safe_dist_mfs, self.mode_avg_threat_mfs)
        self.threat_params = build_tsk_params(self.threat_dist_mfs, self.threat_closure_mfs)
        self.off_thrust_params = build_tsk_params(self.off_thrust_dist_mfs, self.off_thrust_threat_mfs)
        self.def_urg_params = build_tsk_params(self.def_urg_safe_dist_mfs, self.def_urg_max_threat_mfs)
        self.def_steer_params = build_tsk_params(self.def_steer_ang_err_mfs, self.def_steer_urg_mfs)
        self.def_thrust_params = build_tsk_params(self.def_thrust_urg_mfs, self.def_thrust_safe_dist_mfs)

        # 4) Behavior thresholds (5 genes)
        # defensive shooting thresholds
        shoot_dist_raw, shoot_thr_raw = grab(2)
        # distance threshold in [50, 600]
        self.shoot_dist_threshold = 50.0 + shoot_dist_raw * 550.0
        # threat threshold in [0, 1]
        self.shoot_threat_threshold = shoot_thr_raw

        # respawn timing: total duration plus two stage times
        resp_total_raw, resp_t1_raw, resp_t2_raw = grab(3)
        self.respawn_total_time = 1.0 + resp_total_raw * 4.0  # 1 to 5 seconds
        t1_frac = min(resp_t1_raw, resp_t2_raw)
        t2_frac = max(resp_t1_raw, resp_t2_raw)
        self.respawn_stage1_time = t1_frac * self.respawn_total_time
        self.respawn_stage2_time = t2_frac * self.respawn_total_time

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _normalize_safe_distance(self, d: float) -> float:
        if self.max_safe_distance <= 0.0:
            return 0.0
        x = max(0.0, min(d / self.max_safe_distance, 1.0))
        return x

    def _normalize_danger_distance(self, d: float) -> float:
        if self.max_threat_distance <= 0.0:
            return 0.0
        x = 1.0 - max(0.0, min(d / self.max_threat_distance, 1.0))
        return x

    def _normalize_closure(self, closure: float) -> float:
        if self.max_closure_mag <= 0.0:
            return 0.0
        c = max(0.0, min(closure, self.max_closure_mag))
        return c / self.max_closure_mag

    def _compute_threat(self, danger_dist_norm: float, closure_norm: float) -> float:
        return ft.tsk_inference(
            danger_dist_norm,
            closure_norm,
            self.threat_dist_mfs,
            self.threat_closure_mfs,
            self.threat_params,
        )

    def _compute_mode_safety(self, safe_dist_norm: float, avg_threat_norm: float) -> float:
        return ft.tsk_inference(
            safe_dist_norm,
            avg_threat_norm,
            self.mode_safe_dist_mfs,
            self.mode_avg_threat_mfs,
            self.mode_params,
        )

    def _compute_offensive_thrust(self, target_dist_norm: float, avg_threat_norm: float) -> float:
        return ft.tsk_inference(
            target_dist_norm,
            avg_threat_norm,
            self.off_thrust_dist_mfs,
            self.off_thrust_threat_mfs,
            self.off_thrust_params,
        )

    def _compute_def_urgency(self, safe_dist_norm: float, max_threat_norm: float) -> float:
        return ft.tsk_inference(
            safe_dist_norm,
            max_threat_norm,
            self.def_urg_safe_dist_mfs,
            self.def_urg_max_threat_mfs,
            self.def_urg_params,
        )

    def _compute_def_steer_level(self, angle_error_norm: float, escape_urgency: float) -> float:
        return ft.tsk_inference(
            angle_error_norm,
            escape_urgency,
            self.def_steer_ang_err_mfs,
            self.def_steer_urg_mfs,
            self.def_steer_params,
        )

    def _compute_def_thrust_level(self, escape_urgency: float, safe_dist_norm: float) -> float:
        return ft.tsk_inference(
            escape_urgency,
            safe_dist_norm,
            self.def_thrust_urg_mfs,
            self.def_thrust_safe_dist_mfs,
            self.def_thrust_params,
        )

    # ------------------------------------------------------------------
    # Main actions method
    # ------------------------------------------------------------------

    def actions(
        self,
        chromosome,
        ship_state: "ShipOwnState",
        game_state: "GameState",
    ) -> "ActionsReturn":
        """
        Returns (thrust, turn_angle, shoot, mine).
        """

        if game_state["time"] == 0:
            self._reset_run_state()

        self.frame_index += 1

        dt = game_state["delta_time"]
        asteroids = game_state["asteroids"]
        can_shoot = ship_state["can_fire"]

        # Build fuzzy systems and scaling from chromosome each tick
        self._build_fuzzy_from_chromosome(chromosome)

        # Respawn timer maintenance
        if ship_state["is_respawning"]:
            if self.respawn_time <= 0.0:
                self.respawn_time = self.respawn_total_time
            self.respawn_time = max(0.0, self.respawn_time - dt)
        else:
            self.respawn_time = 0.0

        # If no asteroids, log and idle
        if not asteroids:
            self._log_frame(
                ship_state=ship_state,
                game_state=game_state,
                asteroids=[],
                tracked_asteroids={},
                world_positions=[],
                velocities=[],
                mode=self.mode,
                safety_score=self._mode_safety_score,
            )
            return 0.0, 0.0, False, False

        # Persistent IDs and tracking
        tracked_asteroids, ids_unsorted, world_positions, velocities = (
            self._assign_persistent_ids(asteroids, dt)
        )

        # Relative positions
        rel_positions = vm.game_to_ship_frame(
            ship_state["position"],
            world_positions,
            game_state["map_size"],
        )

        # Build structured asteroid data and sort by distance
        ast_data = []
        _hypot = math.hypot
        for aid, rpos, vel, wpos, raw in zip(
            tracked_asteroids.keys(),
            rel_positions,
            velocities,
            world_positions,
            asteroids,
        ):
            dist = _hypot(rpos[0], rpos[1])
            ast_data.append(
                {
                    "id": aid,
                    "rel_pos": rpos,
                    "world_pos": wpos,
                    "vel": vel,
                    "distance": dist,
                    "radius": raw.get("radius", 1.0),
                }
            )

        ast_data.sort(key=lambda a: a["distance"])

        # Per asteroid threat computation
        threats = []
        total_threat = 0.0
        max_threat = 0.0
        min_distance = float("inf")

        for a in ast_data:
            d = a["distance"]
            min_distance = min(min_distance, d)

            danger_dist_norm = self._normalize_danger_distance(d)

            closure = vm.calculate_closure_rate(
                ship_state["position"],
                ship_state["heading"],
                ship_state["speed"],
                a["rel_pos"],
                a["vel"],
            )
            closure_norm = self._normalize_closure(closure)

            thr = self._compute_threat(danger_dist_norm, closure_norm)
            threats.append(thr)
            total_threat += thr
            max_threat = max(max_threat, thr)

        valid_count = len(ast_data)
        avg_threat = total_threat / valid_count if valid_count > 0 else 0.0

        safe_dist_norm = self._normalize_safe_distance(min_distance)
        avg_threat_norm = max(0.0, min(avg_threat, 1.0))
        max_threat_norm = max(0.0, min(max_threat, 1.0))

        # Mode selection with GA tuned hysteresis
        safety_score = self._compute_mode_safety(safe_dist_norm, avg_threat_norm)
        safety_score = max(0.0, min(safety_score, 1.0))
        self._mode_safety_score = safety_score

        if self.mode == "Offensive":
            if safety_score < self.mode_exit_threshold:
                self.mode = "Defensive"
        else:
            if safety_score > self.mode_enter_threshold:
                self.mode = "Offensive"

        # Default actions
        thrust = 0.0
        turn_angle = 0.0
        shoot = False
        mine = False

        # Offensive mode behavior
        if self.mode == "Offensive" and valid_count > 0:

            # Choose highest threat asteroid as target
            best_idx = max(range(valid_count), key=lambda i: threats[i])
            target = ast_data[best_idx]

            # Compute aim using kinematic helper
            ta, on_target = vm.turn_angle(
                ship_state["position"],
                ship_state["heading"],
                ship_state["turn_rate_range"],
                self.bullet_speed,
                target["world_pos"],
                target["vel"],
                dt,
            )
            turn_angle = ta * self.turn_gain

            # Offensive thrust from distance to target and avg threat
            target_dist_norm = self._normalize_safe_distance(target["distance"])
            off_thrust_level = self._compute_offensive_thrust(
                target_dist_norm,
                avg_threat_norm,
            )
            off_thrust_level = max(0.0, min(off_thrust_level, 1.0))
            thrust = off_thrust_level * 100.0 * self.off_thrust_gain

            # Shooting decision
            if on_target and can_shoot:
                shoot = True
                self.asteroids_shot_at.append(target["id"])

        # Defensive mode behavior
        elif self.mode == "Defensive" and valid_count > 0:
            # Compute escape gap
            bearings = [
                vm.heading_relative_angle(
                    [0.0, 0.0],
                    ship_state["heading"],
                    a["rel_pos"],
                ) / 360.0
                for a in ast_data
            ]
            gap_center = vm.largest_gap_center(bearings)

            # Express gap_center as a direction vector in ship frame
            gap_angle_rad = 2.0 * math.pi * gap_center
            gap_vec = [math.cos(gap_angle_rad), math.sin(gap_angle_rad)]

            # Angle from ship heading to gap center in degrees
            angle_error = vm.heading_relative_angle(
                [0.0, 0.0],
                ship_state["heading"],
                gap_vec,
            )

            # Normalize error to [0,1] with 0 = aligned, 1 = 180 deg off
            angle_error_norm = max(0.0, min(abs(angle_error) / 180.0, 1.0))

            # Escape urgency from safe distance and max threat
            escape_urgency = self._compute_def_urgency(safe_dist_norm, max_threat_norm)
            escape_urgency = max(0.0, min(escape_urgency, 1.0))

            # Steering strength
            steer_level = self._compute_def_steer_level(angle_error_norm, escape_urgency)
            steer_level = max(0.0, min(steer_level, 1.0))

            steer_direction = -1.0 if angle_error < 0 else 1.0
            turn_range = ship_state["turn_rate_range"][1]
            turn_angle = steer_direction * steer_level * turn_range * self.turn_gain

            # Thrust level from urgency and distance
            def_thrust_level = self._compute_def_thrust_level(escape_urgency, safe_dist_norm)
            def_thrust_level = max(0.0, min(def_thrust_level, 1.0))
            thrust = def_thrust_level * 100.0 * self.def_thrust_gain

            # Defensive shooting
            closest = ast_data[0]
            closest_threat = threats[0] if threats else 0.0
            if (
                closest["distance"] < self.shoot_dist_threshold
                and closest_threat > self.shoot_threat_threshold
                and can_shoot
            ):
                ta, on_target = vm.turn_angle(
                    ship_state["position"],
                    ship_state["heading"],
                    ship_state["turn_rate_range"],
                    self.bullet_speed,
                    closest["world_pos"],
                    closest["vel"],
                    dt,
                )
                turn_angle = ta * self.turn_gain
                if on_target:
                    shoot = True
                    self.asteroids_shot_at.append(closest["id"])

        # Respawn overrides using GA tuned times
        if ship_state["is_respawning"]:
            if self.respawn_time > self.respawn_stage2_time:
                # earliest stage: blast straight ahead
                self._log_frame(
                    ship_state=ship_state,
                    game_state=game_state,
                    asteroids=asteroids,
                    tracked_asteroids=tracked_asteroids,
                    world_positions=world_positions,
                    velocities=velocities,
                    mode=self.mode,
                    safety_score=self._mode_safety_score,
                )
                return 100.0, 0.0, False, False
            if self.respawn_time > self.respawn_stage1_time:
                # middle stage: rotate but do not thrust
                self._log_frame(
                    ship_state=ship_state,
                    game_state=game_state,
                    asteroids=asteroids,
                    tracked_asteroids=tracked_asteroids,
                    world_positions=world_positions,
                    velocities=velocities,
                    mode=self.mode,
                    safety_score=self._mode_safety_score,
                )
                return 0.0, turn_angle, False, False

        # Log this frame with final chosen mode and safety score
        self._log_frame(
            ship_state=ship_state,
            game_state=game_state,
            asteroids=asteroids,
            tracked_asteroids=tracked_asteroids,
            world_positions=world_positions,
            velocities=velocities,
            mode=self.mode,
            safety_score=self._mode_safety_score,
        )

        return thrust, turn_angle, shoot, mine
