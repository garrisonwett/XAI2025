
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
    A fuzzy-logic Asteroids controller with persistent-per-asteroid IDs.
    Tracks which asteroids you’ve already shot by custom ID so removals
    never desynchronize your list.
    """

    def __init__(self):
        super().__init__()
        self._name = "BajaBlasteroids"

        # --- Mode & cooldown ---
        self.mode = "Avoidance"
        self.switch_tracker = 0

        # --- Shot-tracking by ID ---
        self.asteroids_shot_at: list[int] = []

        # --- Persistent ID machinery ---
        self._tracked_asteroids: dict[int, tuple[float, float]] = {}
        self._next_asteroid_id = 0

        # --- Respawn timer (3s countdown) ---
        self.respawn_time = 0.0

        self.second_tracker = 0

        # --- Bullet speed constant ---
        self.bullet_speed = get_bullet_speed()

    @property
    def name(self) -> str:
        return self._name

    def explanation(self) -> str:
        return getattr(self, "msg", "")

    def actions(
        self,
        ship_state: "ShipOwnState",
        game_state: "GameState",
    ) -> "ActionsReturn":
        """
        Returns (thrust, turn_angle, shoot, mine).
        """



        if game_state["time"] == 0:
            # --- Mode & cooldown ---
            self.mode = "Avoidance"
            self.switch_tracker = 0

            # --- Shot-tracking by ID ---
            self.asteroids_shot_at: list[int] = []

            # --- Persistent ID machinery ---
            self._tracked_asteroids: dict[int, tuple[float, float]] = {}
            self._next_asteroid_id = 0

            # --- Respawn timer (3s countdown) ---
            self.respawn_time = 0.0

            self.second_tracker = 0

            # --- Bullet speed constant ---
            self.bullet_speed = get_bullet_speed()


        EPS = 1e-6
        thrust = EPS
        turn_angle = EPS
        shoot = False

        asteroids = game_state["asteroids"]
        dt = game_state["delta_time"]

        self.second_tracker += dt

        # --- If no asteroids at all, reset target list & bail ---
        if not asteroids:
            self.asteroids_shot_at.clear()
            self._tracked_asteroids.clear()
            return thrust, turn_angle, False, False

        # --- Helpers & caches ---
        _hypot = math.hypot
        _calc_closure = vm.calculate_closure_rate
        _heading_rel = vm.heading_relative_angle
        _tsk = ft.tsk_inference_const

        # --- Default chromosome ---

        chromosome = np.array([0.7109474609320601, 0.7609740700916316, 0.8, 0.3, 0.3717061466030136, 0.9563489000722659, 0.3987398409735582, 0.0, 0.22104530222719243, 0.4717704519369058, 0.9, 0.18773654259862715, 0.6474969600847753, 0.5354084622540757, 0.8140706896832727, 0.7, 0.38268600220069104, 0.7063770643073141, 0.26557552486399494, 0.31088819632944154, 0.3001507138382199, 0.014066650570101369, 0.5701122749556755, 0.09919668307060692, 0.4892249929424314, 0.3798114168134762, 0.221524079282726, 0.7, 0.8330280547072464, 0.0, 0.7727826183648449, 0.5777446510249947, 0.16730090886905546, 0.8796500295328811, 0.4687288848085448, 0.9140098262501138, 0.3, 0.8768659262222392, 0.347900152107138, 0.2052792621001125, 0.7959734436791641, 0.5, 0.08103251576226367, 0.8394081742446953, 0.49063216620197225, 0.6, 0.9634941222800739, 0.37406792745097384, 0.5, 0.3487080894570471, 0.6621192010115009, 0.19167991378593208, 0.3922525883866994, 0.6384264270877891, 0.18240724863601887, 0.7132610842727852, 0.9, 0.3683708522529835, 0.43954176090229546, 0.015674466470348203, 0.3625361595988956, 0.25079722504110336, 0.7995891234915979, 0.2560158477907588, 0.0630307029124193, 0.5321340866123929, 0.9831455370052413, 0.7623128940801136])

        # --- Unpack GA parameters into FIS setups ---
        threat_sum_scalar_1, chromosome = chromosome[0], chromosome[1:]
        thrust_sum_scalar_4, chromosome = chromosome[0], chromosome[1:]

        # FIS1: closure vs distance → threat
        centers, chromosome = chromosome[:1], chromosome[1:]
        closure_mfs_1 = ft.build_triangles(centers)
        centers, chromosome = chromosome[:1], chromosome[1:]
        distance_mfs_1 = ft.build_triangles(centers)
        rule_const_1 = np.array(chromosome[:9]).reshape(
            len(closure_mfs_1), len(distance_mfs_1)
        )
        chromosome = chromosome[9:]

        # FIS2: rel-heading vs size → sub-threat
        centers, chromosome = chromosome[:1], chromosome[1:]
        relative_heading_mfs_2 = ft.build_triangles(centers)
        centers, chromosome = chromosome[:1], chromosome[1:]
        size_mfs_2 = ft.build_triangles(centers)
        rule_const_2 = np.array(chromosome[:9]).reshape(
            len(relative_heading_mfs_2), len(size_mfs_2)
        )
        chromosome = chromosome[9:]

        # FIS3: combine FIS1 & FIS2 → final threat
        centers, chromosome = chromosome[:1], chromosome[1:]
        threat_fis_mfs_1 = ft.build_triangles(centers)
        centers, chromosome = chromosome[:1], chromosome[1:]
        threat_fis_mfs_2 = ft.build_triangles(centers)
        rule_const_3 = np.array(chromosome[:9]).reshape(
            len(threat_fis_mfs_1), len(threat_fis_mfs_2)
        )
        chromosome = chromosome[9:]

        # FIS4: azimuth vs thrust-distance → thrust contrib
        centers, chromosome = chromosome[:1], chromosome[1:]
        az_mfs_4 = ft.build_triangles(centers)
        centers, chromosome = chromosome[:1], chromosome[1:]
        thrust_dist_mfs_4 = ft.build_triangles(centers)
        rule_const_4 = np.array(chromosome[:9]).reshape(
            len(az_mfs_4), len(thrust_dist_mfs_4)
        )
        chromosome = chromosome[9:]

        # FIS5: azimuth vs distance → defensive base
        centers, chromosome = chromosome[:1], chromosome[1:]
        az_mfs_5 = ft.build_triangles(centers)
        centers, chromosome = chromosome[:1], chromosome[1:]
        distance_mfs_5 = ft.build_triangles(centers)
        rule_const_5 = np.array(chromosome[:9]).reshape(
            len(az_mfs_5), len(distance_mfs_5)
        )
        chromosome = chromosome[9:]

        # FIS6: closure vs FIS5 → avoid/shoot decision
        centers, chromosome = chromosome[:1], chromosome[1:]
        relative_heading_mfs_6 = ft.build_triangles(centers)
        centers, chromosome = chromosome[:1], chromosome[1:]
        defensive_fis_mfs_6 = ft.build_triangles(centers)
        rule_const_6 = np.array(chromosome[:9]).reshape(
            len(relative_heading_mfs_6), len(defensive_fis_mfs_6)
        )
        chromosome = chromosome[9:]

        # --- Respawn handling ---
        
        can_shoot = ship_state["can_fire"]

        # --- Build persistent IDs by matching last-frame positions ---
        world_positions = [a["position"] for a in asteroids]
        velocities = [a["velocity"] for a in asteroids]

        new_tracked: dict[int, tuple[float, float]] = {}
        used_old_ids = set()

        for wpos, vel in zip(world_positions, velocities):
            # predict where it was last frame
            pred_x = wpos[0] - vel[0] * dt
            pred_y = wpos[1] - vel[1] * dt
            best_id = None
            best_dist = float("inf")

            # find a previous asteroid whose last pos matches
            for aid, last_pos in self._tracked_asteroids.items():
                if aid in used_old_ids:
                    continue
                d = _hypot(last_pos[0] - pred_x, last_pos[1] - pred_y)
                if d < best_dist:
                    best_dist = d
                    best_id = aid

            # threshold = how far it could have moved + small epsilon
            vel_norm = _hypot(vel[0], vel[1])
            thresh = vel_norm * dt * 1.5 + 1e-3

            if best_id is not None and best_dist <= thresh:
                aid = best_id
            else:
                aid = self._next_asteroid_id
                self._next_asteroid_id += 1

            new_tracked[aid] = wpos
            used_old_ids.add(aid)

        # swap in the new mapping
        self._tracked_asteroids = new_tracked

        # --- Combine into list and sort by distance in ship frame ---
        rel_positions = vm.game_to_ship_frame(
            ship_state["position"], world_positions, game_state["map_size"]
        )
        ast_data = [
            (aid, rpos, vel, wpos, _hypot(rpos[0], rpos[1]))
            for (aid, rpos, vel, wpos) in zip(
                new_tracked.keys(),
                rel_positions,
                velocities,
                world_positions,
            )
        ]
        ast_data.sort(key=lambda x: x[4])  # sort by dist

        # unpack
        (ids_sorted,
         rel_sorted,
         vel_sorted,
         world_sorted,
         dist_sorted) = map(list, zip(*ast_data))

        # --- Compute threat values ---
        threat_array: list[float] = []
        proximity_threat = 0.0

        for i, rpos in enumerate(rel_sorted):
            d = dist_sorted[i]
            d_norm = min(50.0 / (d + EPS), 0.99999)
            closure = _calc_closure(
                ship_state["position"],
                ship_state["heading"],
                ship_state["speed"],
                rpos,
                vel_sorted[i],
            )
            closure = min(max((closure + 200.0) / 400.0, 0.0), 1.0)
            size_n = asteroids[i]["radius"] / 4.0
            rh = _heading_rel([0, 0], ship_state["heading"], rpos) / 360.0
            if rh in (0.0, 1.0):
                rh = 0.99999

            out1 = _tsk(closure, d_norm,
                        closure_mfs_1, distance_mfs_1, rule_const_1)
            out2 = _tsk(rh, size_n,
                        relative_heading_mfs_2, size_mfs_2, rule_const_2)
            thr = _tsk(out1, out2,
                       threat_fis_mfs_1, threat_fis_mfs_2, rule_const_3)

            threat_array.append(thr)
            if d < 400.0:
                proximity_threat += thr

        valid_count = len(threat_array)

        # --- Prune shot list: remove dead IDs, cap oldest off ---
        self.asteroids_shot_at = [
            aid for aid in self.asteroids_shot_at if aid in ids_sorted
        ]



        if len(rel_sorted) == 1 and self.second_tracker%1 < dt:
            self.asteroids_shot_at.clear()

        max_keep = min(20, 4 + valid_count // 2)
        while len(self.asteroids_shot_at) > max_keep:
            self.asteroids_shot_at.pop(0)

        # --- Mode switch with cooldown ---
        if self.switch_tracker <= 0:
            self.mode = (
                "Defensive"
                if proximity_threat > 20.0 * threat_sum_scalar_1
                else "Offensive"
            )
            self.switch_tracker = 30
        else:
            self.switch_tracker -= 1

        # --- OFFENSIVE mode: aim & shoot + thrust-away ---
        if self.mode == "Offensive" and valid_count > 0:
            # choose highest-threat not-yet-shot ID
            shot_set = set(self.asteroids_shot_at)
            for aid, thr in sorted(
                zip(ids_sorted, threat_array),
                key=lambda x: x[1], reverse=True
            ):
                if aid not in shot_set:
                    target_id = aid
                    break
            else:
                target_id = None

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

            # thrust away from any close ones
            for i, rpos in enumerate(rel_sorted):
                if dist_sorted[i] > 300.0:
                    break
                rh = _heading_rel([0, 0], ship_state["heading"], rpos) / 360.0
                if rh in (0.0, 1.0):
                    rh = 0.99999
                dn = min(50.0 / (dist_sorted[i] + EPS), 0.99999)
                thrust += (
                    _tsk(rh, dn, az_mfs_4, thrust_dist_mfs_4, rule_const_4)
                    - 0.5
                )
            thrust *= 200.0 * thrust_sum_scalar_4

        # --- DEFENSIVE mode: avoid or fallback to shooting ---
        elif self.mode == "Defensive" and valid_count > 0:
            avoid_scores = []
            for i, rpos in enumerate(rel_sorted):
                if dist_sorted[i] > 400.0:
                    break
                dn = min(50.0 / (dist_sorted[i] + EPS), 0.99999)
                closure = _calc_closure(
                    ship_state["position"],
                    ship_state["heading"],
                    ship_state["speed"],
                    rpos,
                    vel_sorted[i],
                )
                closure = min(max((closure + 200.0) / 400.0, 0.0), 1.0)
                rh = _heading_rel([0, 0], ship_state["heading"], rpos) / 360.0
                if rh in (0.0, 1.0):
                    rh = 0.99999

                d1 = _tsk(closure, dn, az_mfs_5, distance_mfs_5, rule_const_5)
                d2 = _tsk(rh, d1, relative_heading_mfs_6, defensive_fis_mfs_6, rule_const_6)
                avoid_scores.append(d2)

            if avoid_scores and max(avoid_scores) > 0.5:
                gap = vm.largest_gap_center([
                    _heading_rel([0, 0], ship_state["heading"], r) / 360.0
                    for r in rel_sorted
                ])
                ta, _ = vm.go_to_angle(
                    ship_state["heading"],
                    ship_state["turn_rate_range"],
                    gap,
                    dt,
                )
                turn_angle = ta

            else:
                # fallback to Offensive shooting logic
                shot_set = set(self.asteroids_shot_at)
                for aid, thr in sorted(
                    zip(ids_sorted, threat_array),
                    key=lambda x: x[1], reverse=True
                ):
                    if aid not in shot_set:
                        target_id = aid
                        break
                else:
                    target_id = None

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

                # same thrust-away as Offensive
                for i, rpos in enumerate(rel_sorted):
                    if dist_sorted[i] > 300.0:
                        break
                    rh = _heading_rel([0, 0], ship_state["heading"], rpos) / 360.0
                    if rh in (0.0, 1.0):
                        rh = 0.99999
                    dn = min(50.0 / (dist_sorted[i] + EPS), 0.99999)
                    thrust += (
                        _tsk(rh, dn, az_mfs_4, thrust_dist_mfs_4, rule_const_4)
                        - 0.5
                    )
                thrust *= 200.0 * thrust_sum_scalar_4



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