"""Evaluation loader for RAMBO: the SAC policy on observations normalised with the dataset statistics."""
from baselines.mopo.policy import SACPolicy


class RAMBOPolicy(SACPolicy):
    """The checkpoint carries the normalisation; loading is the same as for MOPO."""
