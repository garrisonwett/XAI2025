from kesslergame import KesslerController
from typing import Dict, Tuple
import numpy as np
import math

# from src.aim_utils.target_new import FuzzyThreatAssess
from src.aim_utils.targeting import FuzzyThreatAssess
from src.vector_utils.trajectories import (
    find_desired_angle,
    find_relative_heading,
    game_to_ship_frame,
)
from src.vector_utils.turn_rate import angle_to_speed, speed_to_thrust
from src.aim_utils.aim_utils import distance_to, closing_rate
from src.avoid_utils.heatmap import generate_heatmap, ast_size_directionality
from src.avoid_utils.avoidance import AvoidThreatAssess, future_position, mid_gap_angles


class FuzzyController(KesslerController):
    def __init__(self) -> None:
        super().__init__()
        self.shot_at = 0
        # Removed the chromosome field so Thales people dont have to worry
        chromosome = [2, 2, 2, 0, 1, 1, 0, 0, 1, 2, 2, 2, 0, 1, 1, 0, 0, 1]
        self.ThreatAssess = FuzzyThreatAssess(chromosome[:9])
        self.AvoidAssess = AvoidThreatAssess(chromosome[9:18])
        self.w_thrust = 400000000
        self.w_angle_diff = 0.3
        self.w_size = 0.8
        self.avoid_exp = 150
        self.max_turn_prev = False
        self.controller = "aggro_shooting"
        self.msg = "Starting the game"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        {'is_respawning': False, 'position': (400.0, 600.0), 'velocity': (-0.0, 0.0), 'speed': 0.0, 'heading': 171.66666666666652, 'mass': 300.0, 'radius': 20.0, 'id': 2, 'team': '2', 'lives_remaining': 3, 'bullets_remaining': -1, 'can_fire': False, 'fire_rate': 10.0, 'thrust_range': (-480.0, 480.0), 'turn_rate_range': (-180.0, 180.0), 'max_speed': 240, 'drag': 80.0}
        {'asteroids': [{'position': (620.4302117728498, 702.8383917354697), 'velocity': (0.8756398609277658, 3.574525552327687), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (513.3505583626527, 263.5901267450178), 'velocity': (-9.581290798376124, 24.23885310919442), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (690.7050200461475, 672.3678891369158), 'velocity': (9.61531839559969, -51.81557807944059), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (999.5925359275391, 626.6535485708916), 'velocity': (-11.882120860690344, -10.62027638516725), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (900.9517123356154, 575.0456786562587), 'velocity': (-26.96833938635827, -16.50264572065669), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (671.7513393447185, 115.27298117728235), 'velocity': (1.684493476357944, 48.53447827180541), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (605.9443246472242, 357.6258501389666), 'velocity': (-1.8708216445566512, -0.8413162414483444), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (525.0179102657784, 38.05102001312225), 'velocity': (34.29667975455838, 10.439400008034024), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (40.120199589149976, 35.998259479225545), 'velocity': (-12.783551271948888, 7.345873150546248), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}, {'position': (932.9462231560831, 778.1358915243183), 'velocity': (-45.33904704729732, -7.263739883070669), 'size': 4, 'mass': 804.247719318987, 'radius': 32.0}], 'ships': [{'is_respawning': False, 'position': (400.0, 400.0), 'velocity': (-0.0, 0.0), 'speed': 0.0, 'heading': 171.66666666666652, 'mass': 300.0, 'radius': 20.0, 'id': 1, 'team': '1', 'lives_remaining': 3}, {'is_respawning': False, 'position': (400.0, 600.0), 'velocity': (-0.0, 0.0), 'speed': 0.0, 'heading': 171.66666666666652, 'mass': 300.0, 'radius': 20.0, 'id': 2, 'team': '2', 'lives_remaining': 3}], 'bullets': [{'position': (16.977778440510995, 721.3938048432695), 'velocity': (-612.8355544951826, 514.2300877492313), 'heading': 140.00000000000003, 'mass': 1.0}, {'position': (87.92388677310007, 592.491412789339), 'velocity': (-680.8933379495996, 419.9812642676485), 'heading': 148.33333333333331, 'mass': 1.0}, {'position': (87.92388677310007, 792.491412789339), 'velocity': (-680.8933379495996, 419.9812642676485), 'heading': 148.33333333333331, 'mass': 1.0}, {'position': (185.74957506126944, 492.41861207580365), 'velocity': (-734.5728855042188, 316.8638128313264), 'heading': 156.6666666666666, 'mass': 1.0}, {'position': (185.74957506126944, 692.4186120758031), 'velocity': (-734.5728855042188, 316.8638128313264), 'heading': 156.6666666666666, 'mass': 1.0}, {'position': (303.40741737109323, 425.8819045102524), 'velocity': (-772.7406610312541, 207.05523608201855), 'heading': 164.9999999999999, 'mass': 1.0}, {'position': (303.40741737109323, 625.8819045102524), 'velocity': (-772.7406610312541, 207.05523608201855), 'heading': 164.9999999999999, 'mass': 1.0}], 'map_size': (1000, 800), 'time': 1.633333333333335, 'time_step': 49}
        """

        ship_position = ship_state["position"]
        ship_heading = ship_state["heading"]
        ship_velocity = ship_state["velocity"]
        speed = 0
        time_to_kill = 0
        rel_heads_gap = []
        use_rel_head_controller = 0
        len_ast = len(game_state["asteroids"])
        num_ast = np.zeros(len_ast)
        tot_size = sum(
            [
                (1 + 3 * (3 ** (asteroid["size"] - 1) - 1) / 2)
                for asteroid in game_state["asteroids"]
            ]
        )

        shoot_threat = num_ast
        avoid_threat = num_ast
        relative_heading = num_ast

        i = 0
        for asteroid in game_state["asteroids"]:
            ast_pos = asteroid["position"]
            ast_vel = asteroid["velocity"]
            ast_size = asteroid["size"]
            dist_to = distance_to(ship_position, ast_pos)

            closure_rate = closing_rate(
                ship_position, ship_velocity, ast_pos, ast_vel
            )
            if dist_to < 2000:
                desired_angle = find_desired_angle(
                    ship_position,
                    800,
                    ast_pos,
                    ast_vel,
                    True,
                )
                angle_diff = abs(desired_angle - ship_heading) / 180
                while angle_diff > 1:
                    angle_diff += -1
                shoot_threat[i] = (
                    self.ThreatAssess.assign_fuzzy_threat(dist_to, closure_rate)
                    + (1 + np.exp(-angle_diff) * self.w_angle_diff)
                    + (1 + np.exp(-ast_size / 4) * self.w_size)
                )
            new_ast_pos = game_to_ship_frame(ship_position, ast_pos)
            new_dist_to = distance_to(ship_position, new_ast_pos)
            if new_dist_to < 1000:
                relative_heading = find_relative_heading(
                    ship_position, new_ast_pos, ship_heading
                )
                avoid_threat = (
                    self.AvoidAssess.assign_fuzzy_threat(new_dist_to, closure_rate)
                    ** self.avoid_exp
                )
                if 90 < relative_heading < 270:
                    speed += self.w_thrust * avoid_threat  # * (-0.5 + ast_size)/4
                else:
                    speed += self.w_thrust * -avoid_threat# * (-0.5 + ast_size)/4
                if new_dist_to < 100:
                    rel_heads_gap.append(relative_heading)
                    use_rel_head_controller += 1
            i += 1

        controller = self.controller
        if (0 < ship_state["bullets_remaining"] < tot_size and self.shot_at > 0) or ship_state["is_respawning"]:
            if ship_state["is_respawning"]:
                self.controller = "respawning_avoider"
            else:
                self.controller = "ammo_saver"
            fire = False
            if len(rel_heads_gap) > 0:
                rel_angle = mid_gap_angles(np.asarray(rel_heads_gap))
                if 90 < rel_angle < 270: rel_angle += -180
                desired_angle = rel_angle + ship_heading
                if desired_angle > 360: desired_angle += -360
                #print(desired_angle)
        elif use_rel_head_controller > 3 and False:
            self.controller = "avoiding_only"
            fire = False
            if len(rel_heads_gap) > 0:
                rel_angle = mid_gap_angles(np.asarray(rel_heads_gap))
                if 90 < rel_angle < 270: rel_angle += -180
                desired_angle = rel_angle + ship_heading
                if desired_angle > 360: desired_angle += -360
                #print(desired_angle)
        else:
            self.controller = "aggro_shooting"
            shoot_idx = np.argmax(shoot_threat)
            ast_pos = game_state["asteroids"][shoot_idx]["position"]
            ast_vel = game_state["asteroids"][shoot_idx]["velocity"]
            desired_angle = find_desired_angle(
                ship_position,
                800,
                ast_pos,
                ast_vel,
                True,
            )
            fire = True
            bullet_closure = closing_rate(
                ship_position, [800*np.cos(np.pi*ship_heading/180), 800*np.sin(np.pi*ship_heading/180)], asteroid["position"], asteroid["velocity"]
            )
            time_to_kill = math.floor(
                distance_to(ship_position, asteroid["position"])*30/bullet_closure
            )
            if controller == "ammo_saver":
                self.msg = (f"Previous asteroid was killed now shooting at {shoot_idx} with time to kill of {time_to_kill}")

        thrust = speed_to_thrust(speed, ship_state["speed"])
        turn_rate = angle_to_speed(desired_angle, ship_heading)
        if abs(thrust) > 480:
            thrust = 480 * thrust / abs(thrust)
        if abs(turn_rate) > 180:
            fire = False
            turn_rate = 180 * turn_rate / abs(turn_rate)
            if self.controller == "aggro_shooting":
                self.max_turn_prev = True
            else:
                self.max_turn_prev = False
        else:
            self.max_turn_prev = False
        self.shot_at += -1

        if ship_state["can_fire"] and fire is True:
            self.shot_at = time_to_kill
        if controller != self.controller:
            self.msg = (f"New controller [{self.controller}] used this time step!")

        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "BajaBlasteroids"
    
    def explanation(self) -> str:
        # Just returns the most recent message. Ideally they would call this whenever self.msg is updated
        return self.msg
