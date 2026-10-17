from .manipulator import Manipulator
from .single_arm import SingleArm

from robustrllib.gym.envs.robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
ROBOT_CLASS_MAPPING = {
    "Kinova3": SingleArm,
}
