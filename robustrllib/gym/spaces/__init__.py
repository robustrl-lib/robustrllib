"""This module implements various spaces.

Spaces describe mathematical sets and are used in Gym to specify valid actions and observations.
Every Gym environment must have the attributes ``action_space`` and ``observation_space``.
If, for instance, three possible actions (0,1,2) can be performed in your environment and observations
are vectors in the two-dimensional unit cube, the environment code may contain the following two lines::

    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(0, 1, shape=(2,))

All spaces inherit from the :class:`Space` superclass.
"""
from robustrllib.gym.spaces.box import Box
from robustrllib.gym.spaces.dict import Dict
from robustrllib.gym.spaces.discrete import Discrete
from robustrllib.gym.spaces.graph import Graph, GraphInstance
from robustrllib.gym.spaces.multi_binary import MultiBinary
from robustrllib.gym.spaces.multi_discrete import MultiDiscrete
from robustrllib.gym.spaces.oneof import OneOf
from robustrllib.gym.spaces.sequence import Sequence
from robustrllib.gym.spaces.space import Space
from robustrllib.gym.spaces.text import Text
from robustrllib.gym.spaces.tuple import Tuple
from robustrllib.gym.spaces.utils import flatdim, flatten, flatten_space, unflatten


__all__ = [
    # base space
    "Space",
    # fundamental spaces
    "Box",
    "Discrete",
    "Text",
    "MultiDiscrete",
    "MultiBinary",
    # composite spaces
    "Graph",
    "GraphInstance",
    "Tuple",
    "Sequence",
    "Dict",
    "OneOf",
    # util functions (there are more utility functions in vector/utils/spaces.py)
    "flatdim",
    "flatten_space",
    "flatten",
    "unflatten",
]
