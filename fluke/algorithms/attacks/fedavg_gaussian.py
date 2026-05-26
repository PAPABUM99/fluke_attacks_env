import torch

from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.fedavg import FedAVG
from fluke.client import MaliciousClient

__all__ = ["FedAVGAttack", "FedAVGGaussianMaliciousClient"]


class FedAVGGaussianMaliciousClient(MaliciousClient):
    def add_noise(self, param):
        param.data.add_(torch.randn_like(param.data) * self.malicious_hp.std + self.malicious_hp.mean)


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGGaussianMaliciousClient