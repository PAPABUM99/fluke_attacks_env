"""Implementation of the Multi-Krum

References:
    BLANCHARD, Peva, et al. Machine learning with adversaries: Byzantine tolerant gradient descent. 
    Advances in neural information processing systems, 2017, 30.
    URL: https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
"""

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

__all__ = ["FedAVGAttack", "FedAVGMultiKrumDefenceServer"]


class FedAVGMultiKrumDefenceServer(Server):
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
        self.malicious_perc: float = kwargs['malicious_perc']
        self.multi_krum: bool = kwargs['multi_krum']

    def _krum_decision_function(self, n, f, m, k_neighbors, client_models):
        # 0. Check feasibility conditions
        if n < 2 * f + 3:
            raise ValueError(f"Krum not applicable: n={n} < 2f+3={2 * f + 3}.")

        def flatten_params(model):
            return torch.cat([param.data.flatten() for param in model.parameters()])

        # 1. Get client's updates matrix of dimension 
        #   [n_clients, n_params] -> (delta_i = param_i - param_global)
        flatten_global = flatten_params(self.model)  # global model's params
        flatten_client = torch.stack([flatten_params(cm) for cm in client_models])  # client's params
        deltas = flatten_client - flatten_global    # client's updates
        
        # 2. Get distance matrix between client's updates [n, n]
        #   ||delta_i - delta_j||^2 = ||delta_i||^2 + ||delta_j||^2 - 2 * (delta_i x delta_j)
        norms = (deltas ** 2).sum(dim=1, keepdim=True)
        D = norms + norms.t() - 2 * (deltas @ deltas.t())
        D = torch.clamp(D, min=0.0).fill_diagonal_(float('inf'))    # with 'inf' the d[i,i] are not considered for the next step 

        # 3. Sort distances for each client's updates and take the argmin score
        closest_k = torch.topk(D, k_neighbors, largest=False, dim=1).values
        scores = closest_k.sum(dim=1)
        
        # 4. Select best clients
        best_idxs = torch.topk(scores, m, largest=False).indices
        best_clients = [client_models[idx] for idx in best_idxs]

        return best_clients

    # Per funzionare ha bisogno nel [ALG_CONFIG] sotto server di: malicious_perc: float, multi_krum: bool
    def aggregate(self, eligible, client_models):
        client_models = list(client_models)
        n = len(client_models)
        # f is an upper bound estimate of Byzantine clients per round
        # assuming malicious_perc of eligible clients are adversarial
        f = int(n * self.malicious_perc)
        m = 1 if not self.multi_krum else n - f    # Krum | Multi-Krum
        k_neighbors = n - f - 2

        # Get model parameters and buffers
        model_parameters = dict(self.model.named_parameters())
        model_buffers = dict(self.model.named_buffers())    # Includes running_mean, running_var, etc.

        # Get the best m clients
        best_clients = self._krum_decision_function(n, f, m, k_neighbors, client_models)

        # Get client parameters and buffers
        best_model_params = [dict(c.named_parameters()) for c in best_clients]
        best_model_buffers = [dict(c.named_buffers()) for c in best_clients]

        for key in model_parameters.keys():
            stacked_params = torch.stack([bp[key].data for bp in best_model_params])
            model_parameters[key].data.copy_(stacked_params.mean(dim=0))
        
        for key in model_buffers.keys():
            if "num_batches_tracked" in key:
                continue
            stacked_buffers = torch.stack([bb[key].data for bb in best_model_buffers])
            model_buffers[key].data.copy_(stacked_buffers.mean(dim=0))

        return self.model


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGMinMaxMaliciousClient

    def get_server_class(self):
        return FedAVGMultiKrumDefenceServer