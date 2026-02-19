import inspect
import re

from kesslergame.bullet import Bullet

from utils import LoggerUtility

logger = LoggerUtility().get_logger()


def get_bullet_speed() -> float:
    """
    Gets the current default bullet speed.
    
    NOTE: We return a hardcoded 800.0 because the KesslerGame library 
    is compiled (mypyc), making 'inspect' impossible, and instantiating 
    a dummy Bullet requires complex owner/ship arguments.
    """
    return 800.0