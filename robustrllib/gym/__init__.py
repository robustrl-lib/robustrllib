"""Root `__init__` of the robustrllib module setting the `__all__` of robustrllib modules."""
# isort: skip_file
import sys as _sys

# Environment modules run `import robustrllib.gym as gym` while this package initialises.
_sys.modules["robustrllib"].gym = _sys.modules[__name__]

from robustrllib.gym.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
    RewardWrapper,
)
from robustrllib.gym.spaces.space import Space
from robustrllib.gym.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,
    make_vec,
    VectorizeMode,
    register_envs,
)

# `envs` registers the environments
from robustrllib.gym import envs
from robustrllib.gym import spaces, utils, vector, wrappers, error, logger, functional

__all__ = [
    # core classes
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "ActionWrapper",
    "RewardWrapper",
    "Space",
    # registration
    "make",
    "make_vec",
    "spec",
    "register",
    "registry",
    "VectorizeMode",
    "pprint_registry",
    "register_envs",
    # module folders
    "envs",
    "spaces",
    "utils",
    "vector",
    "wrappers",
    "error",
    "logger",
    "functional",
]
__version__ = "0.1.0"


# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using
#   pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

import os
import sys

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

try:
    from farama_notifications import notifications

    if "robustrllib" in notifications and __version__ in notifications["robustrllib"]:
        print(notifications["robustrllib"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass
