from enum import Enum

class GenerationType(str, Enum):
    DEFAULT = "default"
    EARLY_STOP = "early_stop"
    DELAY = "delay"