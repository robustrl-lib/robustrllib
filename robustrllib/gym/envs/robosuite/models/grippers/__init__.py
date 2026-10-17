from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .robotiq_85_gripper import Robotiq85Gripper
from .null_gripper import NullGripper

GRIPPER_MAPPING = {
    "Robotiq85Gripper": Robotiq85Gripper,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()
