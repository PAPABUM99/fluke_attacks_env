import torch

from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.fedavg import FedAVG
from fluke.client import MaliciousClient

__all__ = ["FedAVGAttack", "FedAVGLaplaceMaliciousClient"]


class FedAVGLaplaceMaliciousClient(MaliciousClient):
    def add_noise(self, param):
        laplace = torch.distributions.Laplace(self.malicious_hp.loc, self.malicious_hp.scale)
        noise = laplace.rsample(param.data.shape).to(param.data.device)
        param.data.add_(noise)


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGLaplaceMaliciousClient