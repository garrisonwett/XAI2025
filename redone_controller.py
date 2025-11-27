from typing import TYPE_CHECKING, List, Tuple, Optional
from kesslergame import KesslerController

from utils import LoggerUtility
from utils.kessler_helpers import get_bullet_speed
from TeamTempNameSubmission import vector_math as vm
from TeamTempNameSubmission import fuzzy_trees as ft

if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState

import time
import math
import numpy as np


class FuzzyController(KesslerController):
    """
    A fuzzy logic Asteroids controller with persistent per asteroid IDs.
    Tracks which asteroids you have already shot by custom ID so removals
    never desynchronize your list.

    Fuzzy inference systems and thresholds are parameterized by a GA chromosome
    and decoded once per chromosome value, then cached.
    """

    EPS = 1e-6

    def __init__(self):
        super().__init__()
        self._name = "BajaBlasteroids"

        # Mode and cooldown
        self.mode = "Avoidance"
        self.switch_tracker = 0

        # Shot tracking by ID
        self.asteroids_shot_at: list[int] = []

        # Persistent ID machinery
        self._tracked_asteroids: dict[int, tuple[float, float]] = {}
        self._next_asteroid_id = 0

        # Respawn timer (3s countdown)
        self.respawn_time = 0.0

        self.second_tracker = 0.0

        # Bullet speed constant
        self.bullet_speed = get_bullet_speed()

        # GA chromosome and FIS cache
        self._default_chromosome = np.array(
            [
                0.7109474609320601,
                0.7609740700916316,
                0.8,
                0.3,
                0.3717061466030136,
                0.9563489000722659,
                0.3987398409735582,
                0.0,
                0.22104530222719243,
                0.4717704519369058,
                0.9,
                0.18773654259862715,
                0.6474969600847753,
                0.5354084622540757,
                0.8140706896832727,
                0.7,
                0.38268600220069104,
                0.7063770643073141,
                0.26557552486399494,
                0.31088819632944154,
                0.3001507138382199,
                0.014066650570101369,
                0.5701122749556755,
                0.09919668307060692,
                0.4892249929424314,
                0.3798114168134762,
                0.221524079282726,
                0.7,
                0.8330280547072464,
                0.0,
                0.7727826183648449,
                0.5777446510249947,
                0.16730090886905546,
                0.8796500295328811,
                0.4687288848085448,
                0.9140098262501138,
                0.3,
                0.8768659262222392,
                0.347900152107138,
                0.2052792621001125,
                0.7959734436791641,
                0.5,
                0.08103251576226367,
                0.8394081742446953,
                0.49063216620197225,
                0.6,
                0.9634941222800739,
                0.37406792745097384,
                0.5,
                0.3487080894570471,
                0.6621192010115009,
                0.19167991378593208,
                0.3922525883866994,
                0.6384264270877891,
                0.18240724863601887,
                0.7132610842727852,
                0.9,
                0.3683708522529835,
                0.43954176090229546,
                0.015674466470348203,
                0.3625361595988956,
                0.25079722504110336,
                0.7995891234915979,
                0.2560158477907588,
                0.0630307029124193,
                0.5321340866123929,
                0.9831455370052413,
                0.7623128940801136,
            ],
            dtype=float,
        )

        self._current_chromosome: Optional[np.ndarray] = None
        self._fis_params: Optional[dict] = None

    @property
    def name(self) -> str:
        return self._name

    def explanation(self) -> str:
        return getattr(self, "msg", "")

    # ------------------------------------------------------------------
    # Helpers: GA and FIS handling
    # ------------------------------------------------------------------

    def _reset_if_new_episode(self, game_state: "GameState") -> None:
        """Reset all per episode state when game time resets to zero."""
        if game_state["time"] != 0:
            return

        self.mode = "Avoidance"
        self.switch_tracker = 0

        self.asteroids_shot_at = []

        self._tracked_asteroids = {}
        self._next_asteroid_id = 0

        self.respawn_time = 0.0
        self.second_tracker = 0.0

        self.bullet_speed = get_bullet_speed()

    def _ensure_fis_from_chromosome(
        self, chromosome: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Ensure FIS parameters are decoded and cached for the current chromosome.
        Returns the chromosome actually used.
        """
        if chromosome is None:
            chromosome = self._default_chromosome
        else:
            chromosome = np.asarray(chromosome, dtype=float)

        if (
            self._current_chromosome is None
            or self._current_chromosome.shape != chromosome.shape
            or not np.array_equal(self._current_chromosome, chromosome)
        ):
            self._current_chromosome = chromosome.copy()
            self._fis_params = self._decode_chromosome(self._current_chromosome)

        return self._current_chromosome

    def _decode_chromosome(self, chromosome: np.ndarray) -> dict:
        """
        Decode chromosome into scalars and FIS parameter sets.

        Layout as implemented here:
          2 scalar genes:
            threat_sum_scalar_1
            thrust_sum_scalar_4

          For each of 6 FIS blocks:
            1 gene for input 1 MF centers parameter
            1 gene for input 2 MF centers parameter
            (len(mfs1) * len(mfs2) * 3) genes for rule constants
              where each rule gets [p0, p1, p2]
        """
        c = chromosome.copy()
        idx = 0

        def take(n: int) -> np.ndarray:
            nonlocal idx
            segment = c[idx : idx + n]
            idx += n
            return segment

        # Scalars
        threat_sum_scalar_1 = float(take(1)[0])
        thrust_sum_scalar_4 = float(take(1)[0])

        def build_fis() -> Tuple[list, list, np.ndarray]:
            # how build_triangles interprets these center parameters
            # is defined in fuzzy_trees
            centers1 = take(1)
            mfs1 = ft.build_triangles(centers1)

            centers2 = take(1)
            mfs2 = ft.build_triangles(centers2)

            n1 = max(1, len(mfs1))
            n2 = max(1, len(mfs2))
            n_rules = n1 * n2

            # three parameters per rule: p0, p1, p2
            n_params_per_rule = 3
            rules_flat = take(n_rules * n_params_per_rule)

            rules = np.array(
                rules_flat,
                dtype=float,
            ).reshape(n1, n2, n_params_per_rule)
            return mfs1, mfs2, rules

        closure_mfs_1, distance_mfs_1, rule_const_1 = build_fis()
        relative_heading_mfs_2, size_mfs_2, rule_const_2 = build_fis()
        threat_fis_mfs_1, threat_fis_mfs_2, rule_const_3 = build_fis()
        az_mfs_4, thrust_dist_mfs_4, rule_const_4 = build_fis()
        az_mfs_5, distance_mfs_5, rule_const_5 = build_fis()
        relative_heading_mfs_6, defensive_fis_mfs_6, rule_const_6 = build_fis()

        return {
            "threat_sum_scalar_1": threat_sum_scalar_1,
            "thrust_sum_scalar_4": thrust_sum_scalar_4,
            "fis1": (closure_mfs_1, distance_mfs_1, rule_const_1),
            "fis2": (relative_heading_mfs_2, size_mfs_2, rule_const_2),
            "fis3": (threat_fis_mfs_1, threat_fis_mfs_2, rule_const_3),
            "fis4": (az_mfs_4, thrust_dist_mfs_4, rule_const_4),
            "fis5": (az_mfs_5, distance_mfs_5, rule_const_5),
            "fis6": (relative_heading_mfs_6, defensive_fis_mfs_6, rule_const_6),
        }

    # ------------------------------------------------------------------
    # Helpers: normalization and tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_distance(d: float) -> float:
        return min(50.0 / (d + FuzzyController.EPS), 0.99999)

    @staticmethod
    def _norm_closure(closure: float) -> float:
        return min(max((closure + 200.0) / 400.0, 0.0), 1.0)

    @staticmethod
    def _norm_radius(radius: float) -> float:
        # Keep same behavior as original: radius / 4.0
        return radius / 4.0

    def _update_tracked_asteroids(
        self,
        world_positions: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        dt: float,
    ) -> None:
        """Update persistent IDs for asteroids based on motion prediction."""
        _hypot = math.hypot

        new_tracked: dict[int, Tuple[float, float]] = {}
        used_old_ids: set[int] = set()

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

    def _compute_threats(
        self,
        closures: List[float],
        d_norms: List[float],
        size_norms: List[float],
        rel_headings: List[float],
        dist_sorted: List[float],
        fis_params: dict,
    ) -> Tuple[List[float], float]:
        """Compute per asteroid threats and proximity threat."""
        _tsk = ft.tsk_inference_const  # or tsk_inference_add if that is what you use

        closure_mfs_1, distance_mfs_1, rule_const_1 = fis_params["fis1"]
        relative_heading_mfs_2, size_mfs_2, rule_const_2 = fis_params["fis2"]
        threat_fis_mfs_1, threat_fis_mfs_2, rule_const_3 = fis_params["fis3"]

        threat_array: List[float] = []
        proximity_threat = 0.0

        for i in range(len(closures)):
            out1 = _tsk(
                closures[i],
                d_norms[i],
                closure_mfs_1,
                distance_mfs_1,
                rule_const_1,
            )
            out2 = _tsk(
                rel_headings[i],
                size_norms[i],
                relative_heading_mfs_2,
                size_mfs_2,
                rule_const_2,
            )
            thr = _tsk(
                out1,
                out2,
                threat_fis_mfs_1,
                threat_fis_mfs_2,
                rule_const_3,
            )
            threat_array.append(thr)
            if dist_sorted[i] < 400.0:
                proximity_threat += thr

        return threat_array, proximity_threat

    def _update_mode(self, proximity_threat: float, fis_params: dict) -> None:
        """Handle mode switching with cooldown."""
        threat_sum_scalar_1 = fis_params["threat_sum_scalar_1"]
        if self.switch_tracker <= 0:
            self.mode = (
                "Defensive"
                if proximity_threat > 20.0 * threat_sum_scalar_1
                else "Offensive"
            )
            self.switch_tracker = 30
        else:
            self.switch_tracker -= 1

    @staticmethod
    def _select_target_id(
        ids_sorted: List[int],
        threat_array: List[float],
        shot_set: set[int],
    ) -> Optional[int]:
        """Select the most threatening asteroid ID that has not been shot at."""
        for aid, thr in sorted(
            zip(ids_sorted, threat_array),
            key=lambda x: x[1],
            reverse=True,
        ):
            if aid not in shot_set:
                return aid
        return None

    def _compute_thrust_away(
        self,
        rel_headings: List[float],
        d_norms: List[float],
        dist_sorted: List[float],
        fis_params: dict,
        base_thrust: float,
    ) -> float:
        """Compute thrust contributions that push away from nearby asteroids."""
        _tsk = ft.tsk_inference_const  # or tsk_inference_add

        az_mfs_4, thrust_dist_mfs_4, rule_const_4 = fis_params["fis4"]

        thrust = base_thrust
        for i, rh in enumerate(rel_headings):
            if dist_sorted[i] > 300.0:
                break
            dn = d_norms[i]
            thrust += (
                _tsk(
                    rh,
                    dn,
                    az_mfs_4,
                    thrust_dist_mfs_4,
                    rule_const_4,
                )
                - 0.5
            )
        thrust *= 200.0 * fis_params["thrust_sum_scalar_4"]
        return thrust

    # ------------------------------------------------------------------
    # Main control interface
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
        self._reset_if_new_episode(game_state)

        thrust = self.EPS
        turn_angle = self.EPS
        shoot = False

        asteroids = game_state["asteroids"]
        dt = game_state["delta_time"]

        self.second_tracker += dt

        # If no asteroids, reset target list and bail
        if not asteroids:
            self.asteroids_shot_at.clear()
            self._tracked_asteroids.clear()
            return thrust, turn_angle, False, False

        # Helper aliases
        _hypot = math.hypot
        _calc_closure = vm.calculate_closure_rate
        _heading_rel = vm.heading_relative_angle

        # Ensure FIS parameters are cached
        chromosome_used = self._ensure_fis_from_chromosome(chromosome)
        fis_params = self._fis_params
        assert fis_params is not None

        # Convenience unpack for FIS 5 and 6 used in defensive logic
        az_mfs_5, distance_mfs_5, rule_const_5 = fis_params["fis5"]
        relative_heading_mfs_6, defensive_fis_mfs_6, rule_const_6 = fis_params[
            "fis6"
        ]

        can_shoot = ship_state["can_fire"]

        # Build persistent IDs
        world_positions = [a["position"] for a in asteroids]
        velocities = [a["velocity"] for a in asteroids]
        radii = [a["radius"] for a in asteroids]

        self._update_tracked_asteroids(world_positions, velocities, dt)

        # Transform to ship frame
        rel_positions = vm.game_to_ship_frame(
            ship_state["position"],
            world_positions,
            game_state["map_size"],
        )

        # Build combined asteroid data and sort by distance
        ast_data = [
            (
                aid,
                rpos,
                vel,
                wpos,
                _hypot(rpos[0], rpos[1]),
                radius,
            )
            for (aid, rpos, vel, wpos, radius) in zip(
                self._tracked_asteroids.keys(),
                rel_positions,
                velocities,
                world_positions,
                radii,
            )
        ]
        ast_data.sort(key=lambda x: x[4])

        (
            ids_sorted,
            rel_sorted,
            vel_sorted,
            world_sorted,
            dist_sorted,
            radius_sorted,
        ) = map(list, zip(*ast_data))

        valid_count = len(ids_sorted)
        if valid_count == 0:
            return thrust, turn_angle, shoot, False

        # Precompute normalized features
        closures: List[float] = []
        d_norms: List[float] = []
        rel_headings: List[float] = []
        size_norms: List[float] = []

        for i, rpos in enumerate(rel_sorted):
            d = dist_sorted[i]
            d_norm = self._norm_distance(d)
            d_norms.append(d_norm)

            closure_raw = _calc_closure(
                ship_state["position"],
                ship_state["heading"],
                ship_state["speed"],
                rpos,
                vel_sorted[i],
            )
            closure = self._norm_closure(closure_raw)
            closures.append(closure)

            rh = _heading_rel(
                [0, 0],
                ship_state["heading"],
                rpos,
            ) / 360.0
            if rh in (0.0, 1.0):
                rh = 0.99999
            rel_headings.append(rh)

            size_n = self._norm_radius(radius_sorted[i])
            size_norms.append(size_n)

        # Compute threats
        threat_array, proximity_threat = self._compute_threats(
            closures,
            d_norms,
            size_norms,
            rel_headings,
            dist_sorted,
            fis_params,
        )

        # Prune shot list: remove dead IDs and cap oldest off
        self.asteroids_shot_at = [
            aid for aid in self.asteroids_shot_at if aid in ids_sorted
        ]

        if len(rel_sorted) == 1 and self.second_tracker % 1 < dt:
            self.asteroids_shot_at.clear()

        max_keep = min(20, 4 + valid_count // 2)
        while len(self.asteroids_shot_at) > max_keep:
            self.asteroids_shot_at.pop(0)

        # Mode switch
        self._update_mode(proximity_threat, fis_params)

        # OFFENSIVE mode: aim and shoot plus thrust away
        if self.mode == "Offensive" and valid_count > 0:
            shot_set = set(self.asteroids_shot_at)
            target_id = self._select_target_id(
                ids_sorted,
                threat_array,
                shot_set,
            )

            if target_id is not None:
                idx = ids_sorted.index(target_id)
                ta, on_target = vm.turn_angle(
                    ship_state["position"],
                    ship_state["heading"],
                    ship_state["turn_rate_range"],
                    self.bullet_speed,
                    world_sorted[idx],
                    vel_sorted[idx],
                    dt,
                )
                turn_angle = ta
                if on_target and can_shoot:
                    shoot = True
                    self.asteroids_shot_at.append(target_id)

            thrust = self._compute_thrust_away(
                rel_headings,
                d_norms,
                dist_sorted,
                fis_params,
                thrust,
            )

        # DEFENSIVE mode: avoid or fallback to shooting
        elif self.mode == "Defensive" and valid_count > 0:
            avoid_scores: List[float] = []

            _tsk = ft.tsk_inference_const  # or tsk_inference_add
            for i in range(valid_count):
                if dist_sorted[i] > 400.0:
                    break

                d1 = _tsk(
                    closures[i],
                    d_norms[i],
                    az_mfs_5,
                    distance_mfs_5,
                    rule_const_5,
                )
                d2 = _tsk(
                    rel_headings[i],
                    d1,
                    relative_heading_mfs_6,
                    defensive_fis_mfs_6,
                    rule_const_6,
                )
                avoid_scores.append(d2)

            if avoid_scores and max(avoid_scores) > 0.5:
                gap = vm.largest_gap_center(rel_headings)
                ta, _ = vm.go_to_angle(
                    ship_state["heading"],
                    ship_state["turn_rate_range"],
                    gap,
                    dt,
                )
                turn_angle = ta
            else:
                # Fallback to Offensive shooting logic
                shot_set = set(self.asteroids_shot_at)
                target_id = self._select_target_id(
                    ids_sorted,
                    threat_array,
                    shot_set,
                )

                if target_id is not None:
                    idx = ids_sorted.index(target_id)
                    ta, on_target = vm.turn_angle(
                        ship_state["position"],
                        ship_state["heading"],
                        ship_state["turn_rate_range"],
                        self.bullet_speed,
                        world_sorted[idx],
                        vel_sorted[idx],
                        dt,
                    )
                    turn_angle = ta
                    if on_target and can_shoot:
                        shoot = True
                        self.asteroids_shot_at.append(target_id)

                thrust = self._compute_thrust_away(
                    rel_headings,
                    d_norms,
                    dist_sorted,
                    fis_params,
                    thrust,
                )

        # Respawn handling overrides
        if ship_state["is_respawning"]:
            if self.respawn_time <= 0.0:
                self.respawn_time = 3.0
            self.respawn_time = max(0.0, self.respawn_time - dt)

            if self.respawn_time > 2.0:
                return 100.0, 0.0, False, False
            if self.respawn_time > 1.0:
                return 0.0, turn_angle, False, False
        else:
            self.respawn_time = 0.0

        return thrust, turn_angle, shoot, False


# Chromosome length notes:
#   This controller always uses the chromosome you pass into actions.
#   If chromosome is None, it falls back to _default_chromosome.
#
#   Genes actually used:
#     2 scalar genes:
#       threat_sum_scalar_1
#       thrust_sum_scalar_4
#     For each of 6 FIS blocks:
#       1 gene for input 1 MF center parameter
#       1 gene for input 2 MF center parameter
#       len(mfs1) * len(mfs2) * 3 genes for rule constants (p0, p1, p2)
#
#   Any extra genes beyond what is consumed are ignored.
