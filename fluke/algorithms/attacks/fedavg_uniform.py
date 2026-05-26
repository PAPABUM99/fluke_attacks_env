import torch

from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.fedavg import FedAVG
from fluke.client import MaliciousClient

__all__ = ["FedAVGAttack", "FedAVGUniformMaliciousClient"]


class FedAVGUniformMaliciousClient(MaliciousClient):
    def add_noise(self, param):
        param.data.add_(torch.empty_like(param.data).uniform_(self.malicious_hp.low, self.malicious_hp.high))


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGUniformMaliciousClient