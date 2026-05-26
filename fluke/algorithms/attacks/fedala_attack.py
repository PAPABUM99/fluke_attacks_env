from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.fedala import FedALA

__all__ = ["FedALAAttack"]


class FedALAAttack(MaliciousMixin, FedALA):
    pass