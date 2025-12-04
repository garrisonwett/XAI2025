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
    """
    A fuzzy-logic Asteroids controller.
    """

    def __init__(self, chromosome):
        super().__init__()
        self._name = "BajaBlasteroids"
        self.chromosome = chromosome
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

        # ---------------------------------
        # Ship info
        # ---------------------------------
        ship_pos = ship_state["position"]
        ship_speed = ship_state["speed"]
        ship_heading = ship_state["heading"]
        turn_rate_range = ship_state["turn_rate_range"]

        # ---------------------------------
        # World info
        # ---------------------------------
        asteroids = game_state["asteroids"]
        delta_time = game_state["delta_time"]

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
