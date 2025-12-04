from typing import TYPE_CHECKING
from kesslergame import KesslerController

import math
import numpy as np

from utils.kessler_helpers import get_bullet_speed
from TeamTempNameSubmission import vector_math as vm

from algorithms import fuzzy_tree_output   # safe: algorithms does NOT import controller

if TYPE_CHECKING:
    from utils.types import ActionsReturn, GameState, ShipOwnState


class FuzzyController(KesslerController):

    def __init__(self, chromosome):
        super().__init__()
        self._name = "BajaBlasteroids"
        self.chromosome = chromosome

        # Import print helpers
        from algorithms import (
            assign_fis_ids,
            print_tree_structure,
            print_membership_functions,
            print_rule_constants
        )

        # # Assign IDs for readability
        # assign_fis_ids(self.chromosome)

        # print("\n================ Fuzzy Tree Structure ================")
        # print_tree_structure(self.chromosome)

        # print("\n================ Membership Functions ================")
        # print_membership_functions(self.chromosome)

        # print("\n================ Rule Constants ================")
        # print_rule_constants(self.chromosome)

        # print("\n================ End Tree Print =================\n")

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
        ship_vel          = ship_state["velocity"]
        ship_speed        = ship_state["speed"]
        ship_heading      = ship_state["heading"]
        ship_mass         = ship_state["mass"]
        ship_radius       = ship_state["radius"]
        ship_id           = ship_state["id"]
        ship_team         = ship_state["team"]
        ship_lives        = ship_state["lives_remaining"]
        ship_respawning   = ship_state["is_respawning"]

        bullets_remaining = ship_state["bullets_remaining"]
        mines_remaining   = ship_state["mines_remaining"]
        can_fire          = ship_state["can_fire"]
        fire_rate         = ship_state["fire_rate"]
        can_deploy_mine   = ship_state["can_deploy_mine"]
        mine_deploy_rate  = ship_state["mine_deploy_rate"]

        thrust_range      = ship_state["thrust_range"]
        turn_rate_range   = ship_state["turn_rate_range"]
        max_speed         = ship_state["max_speed"]
        drag              = ship_state["drag"]

        # -------------------------------
        # Parse world information
        # -------------------------------
        asteroids   = game_state["asteroids"]
        ships       = game_state["ships"]
        bullets     = game_state["bullets"]
        mines       = game_state["mines"]

        map_width, map_height = game_state["map_size"]
        world_time            = game_state["time"]
        delta_time            = game_state["delta_time"]
        frame                 = game_state["sim_frame"]
        time_limit            = game_state["time_limit"]

        # --------------------------------
        # Parsed lists for convenience
        # --------------------------------
        asteroid_positions  = [a["position"] for a in asteroids]
        asteroid_velocities = [a["velocity"] for a in asteroids]
        asteroid_radii      = [a["radius"] for a in asteroids]

        other_ship_positions = [s["position"] for s in ships]
        other_ship_headings  = [s["heading"] for s in ships]

        bullet_positions     = [b["position"] for b in bullets]
        bullet_velocities    = [b["velocity"] for b in bullets]

        mine_positions       = [m["position"] for m in mines]
        mine_remaining_times = [m["remaining_time"] for m in mines]

        # --------------------------------
        # Placeholder outputs for now
        # --------------------------------


        if not asteroids:
            return 0.0, 0.0, False, False

        # ---------------------------------
        # Threat determination
        # ---------------------------------
        highest_threat_value = -9999
        highest_threat_index = 0

        for i, asteroid in enumerate(asteroids):

            apos = asteroid["position"]
            avel = asteroid["velocity"]
            arad = asteroid["radius"]

            # 1. relative heading (0 to 1)
            relative_heading = vm.heading_relative_angle(
                ship_pos,
                ship_heading,
                apos
            ) / 360.0

            # 2. relative asteroid position (dx, dy)
            rel_position = vm.game_to_ship_frame(
                ship_pos,
                [apos],
                game_state["map_size"]
            )[0]

            # 3. closure rate
            closure = vm.calculate_closure_rate(
                ship_pos,
                ship_heading,
                ship_speed,
                apos,
                avel
            )

            # 4. distance
            distance = math.hypot(apos[0] - ship_pos[0], apos[1] - ship_pos[1])

            # Input Scaling
            closure = np.clip((closure + 100) / 200, 0.0, 1.0)
            relative_heading = abs(relative_heading-1)  # Already 0 to 1
            arad = np.clip((arad-8) / 24, 0.0, 1.0)
            distance = np.clip(distance/(np.sqrt(map_height**2 + map_width**2)),0,1)  # Closer = higher input

           # 5. fuzzy output




            

            threat = fuzzy_tree_output(
                self.chromosome,
                relative_heading,
                closure,
                arad,
                distance
            )

            if threat > highest_threat_value:
                highest_threat_value = threat
                highest_threat_index = i

        # ---------------------------------
        # Aim at the highest-threat asteroid
        # ---------------------------------
        target = asteroids[highest_threat_index]
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
        # Simple actions (you can evolve these later)
        # ---------------------------------
        thrust = 0.0
        turn_rate = turn_angle

        if on_target:
            shoot = True
        else:
            shoot = False
            
        deploy_mine = False

        return thrust, turn_rate, shoot, deploy_mine
