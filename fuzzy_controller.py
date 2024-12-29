from kesslergame import KesslerController

from utils.types import ActionsReturn, ShipOwnState, convert_game_state


class FuzzyController(KesslerController):
    """
    The main class for the fuzzy controller.

    Args:
       KesslerController (KesslerController): The base class for the controller. Provided by the kesslergame package.
    """

    def __init__(self):
        """
        Initializes the fuzzy controller.
        All state variables should be initialized here.
        """
        super().__init__()

        self._name: str = "BajaBlasteroids"

        # Ship Variables

        # Threat Variables

    @property
    def name(self) -> str:
        """Getter method for the name of the controller."""
        return self._name

    def control(self, observation):
        # Implement your fuzzy logic here
        return 0

    def explanation(self) -> str:
        # Just returns the most recent message. Ideally they would call this whenever self.msg is updated
        return self.msg

    def actions(self, ship_state, game_state) -> ActionsReturn:
        ship_state = ShipOwnState(**ship_state)
        game_state = convert_game_state(game_state)

        return 0.0, 0.0, True, True
