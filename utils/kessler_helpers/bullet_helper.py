import inspect
import re

from kesslergame.bullet import Bullet

from utils import LoggerUtility

logger = LoggerUtility().get_logger()


def get_bullet_speed() -> float:
    """
    Gets the current default bullet speed from the Kessler Game `Bullet` class.

    Returns:
        float: The default bullet speed.
    """
    try:
        lines = inspect.getsource(Bullet)
        match = re.search(r"self\.speed\s*=\s*([0-9.]+)", lines)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError, TypeError) as e:
        logger.error(f"Error in get_bullet_speed: {e}")
    return 800.0
