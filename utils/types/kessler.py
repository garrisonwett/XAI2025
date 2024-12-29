from typing import NamedTuple, Tuple, TypedDict, List


class AsteroidState(TypedDict):
    """
    The state of an asteroid in the game.

    Attributes:
        position (`Tuple[float, float]`): The position of the asteroid.
        velocity (`Tuple[float, float]`): The velocity of the asteroid.
        size (`int`): The size of the asteroid.
        mass (`float`): The mass of the asteroid.
        radius (`float`): The radius of the asteroid.
    """

    position: Tuple[float, float]
    velocity: Tuple[float, float]
    size: int
    mass: float
    radius: float


class ShipState(TypedDict):
    """
    The state of a ship in the game.

    Attributes:
        is_respawning (`bool`): Whether the ship is respawning.
        position (`Tuple[float, float]`): The position of the ship.
        velocity (`Tuple[float, float]`): The velocity of the ship.
        speed (`float`): The speed of the ship.
        heading (`float`): The heading of the ship.
        mass (`float`): The mass of the ship.
        radius (`float`): The radius of the ship.
        id (`int`): The ID of the ship.
        team (`str`): The team of the ship.
        lives_remaining (`int`): The number of lives remaining for the ship.
    """

    is_respawning: bool
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    speed: float
    heading: float
    mass: float
    radius: float
    id: int
    team: str
    lives_remaining: int


class ShipOwnState(ShipState):
    """
    The state of a ship in the game, including additional information for the ship's own state.

    Attributes:
        bullets_remaining (`int`): The number of bullets remaining for the ship.
        mines_remaining (`int`): The number of mines remaining for the ship.
        can_fire (`bool`): Whether the ship can fire a bullet.
        fire_rate (`float`): The rate at which the ship can fire bullets.
        can_deploy_mine (`bool`): Whether the ship can deploy a mine.
        mine_deploy_rate (`float`): The rate at which the ship can deploy mines.
        thrust_range (`Tuple[float, float]`): The range of thrust values the ship can use.
        turn_rate_range (`Tuple[float, float]`): The range of turn rates the ship can use.
        max_speed (`float`): The maximum speed of the ship.
        drag (`float`): The drag of the ship.

    Examples:
        ```python
        # Get the number of bullets remaining for the ship
        bullets_remaining = ship_own_state["bullets_remaining"]
        ```
    """

    bullets_remaining: int
    mines_remaining: int
    can_fire: bool
    fire_rate: float
    can_deploy_mine: bool
    mine_deploy_rate: float
    thrust_range: Tuple[float, float]
    turn_rate_range: Tuple[float, float]
    max_speed: float
    drag: float


class BulletState(TypedDict):
    """
    The state of a bullet in the game.

    Attributes:
        position (`Tuple[float, float]`): The position of the bullet.
        velocity (`Tuple[float, float]`): The velocity of the bullet.
        heading (`float`): The heading of the bullet.
        mass (`float`): The mass of the bullet.
    """

    position: Tuple[float, float]
    velocity: Tuple[float, float]
    heading: float
    mass: float


class MineState(TypedDict):
    """
    The state of a mine in the game.

    Attributes:
        position (`Tuple[float, float]`): The position of the mine.
        mass (`float`): The mass of the mine.
        fuse_time (`float`): The fuse time of the mine.
        remaining_time (`float`): The remaining time of the mine.
    """

    position: Tuple[float, float]
    mass: float
    fuse_time: float
    remaining_time: float


class GameState(TypedDict):
    """
    The game state for the kessler game.

    Attributes:
        asteroids (`List[AsteroidState]`): The state of all asteroids in the game.
        ships (`List[ShipState]`): The state of all ships in the game.
        bullets (`List[BulletState]`): The state of all bullets in the game.
        mines (`List[MineState]`): The state of all mines in the game.
        map_size (`Tuple[int, int]`): The size of the map.
        time (`float`): The current time in the game.
        delta_time (`float`): The time since the last frame.
        sim_frame (`int`): The current simulation frame.
        time_limit (`float`): The time limit for the game.

    Examples:
        ```python
        # Get all asteroid positions
        asteroid_positions = [asteroid["position"] for asteroid in game_state["asteroids"]]

        # Get all active ships
        active_ships = [ship for ship in game_state["ships"] if not ship["is_respawning"]]
        ```
    """

    asteroids: List[AsteroidState]
    ships: List[ShipState]
    bullets: List[BulletState]
    mines: List[MineState]
    map_size: Tuple[int, int]
    time: float
    delta_time: float
    sim_frame: int
    time_limit: float


class ActionsReturn(NamedTuple):
    """
    The return type for the overloaded actions method.

    Attributes:
        ship_thrust (`float`): The thrust to apply to the ship.
        ship_turn_rate (`float`): The turn rate to apply to the ship.
        fire (`bool`): Whether to fire a bullet.
        deploy_mine (`bool`): Whether to deploy a mine.
    """

    ship_thrust: float
    ship_turn_rate: float
    fire: bool
    deploy_mine: bool
