"""Bridge between the standard Gymnasium API and the environments' own step convention.

Several environments expect ``step({"action": a, "robust_config": args})`` and branch on
``args.noise_factor``. This wrapper lets the library use the plain ``step(action)`` API:
it passes settings under which every in-environment perturbation branch is switched off
(``robustrllib.gym.envs.inert_args``), so all perturbation comes from the shift wrappers
stacked on top of this one.
"""
from __future__ import annotations

import robustrllib.gym as gym
from robustrllib.gym.envs.inert_args import inert_args


def _uses_dict_api(env) -> bool:
    """Detect the step convention by the name of ``step``'s first argument.

    The dict-API environments declare ``def step(self, robust_input)``; the raw-action
    environments (PointMaze, Pusher-v5s, DoorCausal) declare ``def step(self, action)``.
    """
    import inspect

    try:
        params = list(inspect.signature(env.unwrapped.step).parameters)
    except (ValueError, TypeError):
        return True
    return bool(params) and params[0] == "robust_input"


class RobustCoreWrapper(gym.Wrapper):
    """Adapt the ``step(action)`` call to whichever convention the underlying env uses:

    * MuJoCo v5 / Box2D / Fetch / Adroit environments take ``step({"action", "robust_config"})``.
    * PointMaze, Pusher-v5s and DoorCausal take a raw ``action``.
    """

    def __init__(self, env):
        super().__init__(env)
        self._config = inert_args()
        self._dict_api = _uses_dict_api(env)

    def step(self, action):
        if self._dict_api:
            return self.env.step({"action": action, "robust_config": self._config})
        return self.env.step(action)
