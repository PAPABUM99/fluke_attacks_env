from typing import Sequence

import torch

from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.attacks.fedavg_gaussian import FedAVGGaussianMaliciousClient
from fluke.algorithms.attacks.fedavg_laplace import FedAVGLaplaceMaliciousClient
from fluke.algorithms.attacks.fedavg_min_max import FedAVGMinMaxMaliciousClient
from fluke.algorithms.attacks.fedavg_sign_flipping import FedAVGSignFlippingMaliciousClient
from fluke.algorithms.attacks.fedavg_uniform import FedAVGUniformMaliciousClient
from fluke.algorithms.fedavg import FedAVG
from fluke.client import Client
from fluke.data import FastDataLoader
from fluke.server import Server


__all__ = ["FedAVGAttack", "FedAVGTrimmedMeanDefenceServer"]


class FedAVGTrimmedMeanDefenceServer(Server):
    def __init__(
            self, 
            model: torch.nn.Module, 
            test_set: FastDataLoader | None, 
            clients: Sequence[Client], 
            weighted: bool = False, 
            lr: float = 1.0, 
            **kwargs
        ):
        super().__init__(
            model, 
            test_set, 
            clients, 
            weighted, 
            lr,
            **kwargs
        )
        self.trim: float = kwargs['trim']

    # TODO: sistema commenti e letteraturA
    # Per funzionare serve impostare in [ALG_CONFIG] sotto server trim: float
    def aggregate(self, eligible, client_models):
        client_models = list(client_models) # client's model list
        n = len(client_models)  # total normal clients
        f = int(n * self.trim) # total trimmed clients
        
        if 2 * f >= n:
            raise ValueError(f"trim={self.trim} too large: need n > 2f, got n={n}, k={f}.")

        # Get model parameters and buffers
        model_parameters = dict(self.model.named_parameters())
        model_buffers = dict(self.model.named_buffers())    # Includes running_mean, running_var, etc.

        # Get clients parameters and buffers
        client_parameters = [dict(m.named_parameters()) for m in client_models]
        client_buffers = [dict(m.named_buffers()) for m in client_models]
        
        # Compute trimmed mean over all parameters for each client, per layer
        for key in model_parameters.keys():
            stacked_params = torch.stack([cp[key].data for cp in client_parameters])
            sorted_params = stacked_params.sort(dim=0).values
            trimmed_params = sorted_params[f:n - f] # trimming
            model_parameters[key].data.copy_(trimmed_params.mean(dim=0))
        
        for key in model_buffers.keys():
            if "num_batches_tracked" in key:
                continue
            else:
                stacked_buffers = torch.stack([cb[key].data for cb in client_buffers])
                sorted_buffers = stacked_buffers.sort(dim=0).values
                trimmed_buffers = sorted_buffers[f:n - f]   # trimming
                model_buffers[key].data.copy_(trimmed_buffers.mean(dim=0))

        return self.model


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGMinMaxMaliciousClient
    
    def get_server_class(self):
        return FedAVGTrimmedMeanDefenceServer