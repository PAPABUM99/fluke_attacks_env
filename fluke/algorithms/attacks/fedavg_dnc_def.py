""" Implementation of the (Divide-and-Conquer) AGR Algorithm
    
    References:
    Shejwalkar, Virat & Houmansadr, Amir. (2021). 
    Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. 
    10.14722/ndss.2021.24498. 
    URL: https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
"""

# import logging
import random
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
from torch.nn.utils import parameters_to_vector, vector_to_parameters


__all__ = ["FedAVGAttack", "FedAVGDnCDefenseServer"]

# Imposta in [ALG_CONFIG] sotto server 'niters: int = 1', 'c: int = 1', 'b: int = 10000' 'malicious_perc: float'
class FedAVGDnCDefenseServer(Server):
    def __init__(
            self, 
            model: torch.nn.Module, 
            test_set: FastDataLoader | None, 
            clients: Sequence[Client], 
            weighted: bool = False, 
            lr: float = 1.0, 
            **kwargs,
        ):
        super().__init__(
            model, 
            test_set, 
            clients, 
            weighted, 
            lr, 
            **kwargs,
        )
        self.lr = self.hyper_params.lr
        self.mal_perc: float = kwargs["malicious_perc"]
        self.niters: int = kwargs["niters"]
        self.c: float = kwargs["c"]
        self.b: int = kwargs["b"]

    def _dnc_aggregation_rule(self, n, d, m, updates, client_models):
        I_good = [] # set of indices of good updates
        for _ in range(self.niters):
            # 1. Subsampling updates
            # 1.1 Sorted set of indices from d
            r = torch.tensor(random.sample(range(d), self.b)).sort().values
            sub_updates = updates[:, r]

            # 2. Compute mean of the subsampled updates
            mu = sub_updates.mean(dim=0)
            
            # 3. SVD based spectral method
            # 3.1 Centering subsampled updates
            cen_sub_updates = sub_updates - mu
            
            # 3.2 Compute top right singular eigenvector 
            #     from centered subsampled updates via SVD
            _, _, Vt = torch.linalg.svd(cen_sub_updates, full_matrices=False)
            v = Vt[0]
            # alternative method
            # dist = cen_sub_updates @ cen_sub_updates.t()
            # eigvals, eigvecs = torch.linalg.eigh(dist)
            # idx = torch.argmax(eigvals)
            # u = eigvecs[:, idx]
            # v = cen_sub_updates.t() @ u
            # v = v / torch.linalg.norm(v)
            
            # 4. Compute outlier scores
            scores = (cen_sub_updates @ v) ** 2
            
            # 5. Compute set of indices from the lowest outlier scores
            k = n - int(self.c * m) # number of updates to keep
            I = torch.topk(scores, k=k, largest=False).indices
            I_good.append(set(I.tolist()))
        
        # 6. Compute the mean of the benign updates following the set of good indices
        I_final = set.intersection(*I_good) # argument unpacking
        # print("Kept clients:", len(I_final), "Removed:", n - len(I_final))
        if len(I_final) == 0:
            # logging.warning("DnC: I_final is empty, falling back to all updates")
            I_final = set(range(n))
        I_final = torch.tensor(sorted(list(I_final)), dtype=torch.long)

        aggregated_update = updates[I_final].mean(dim=0)
        good_clients = [client_models[i] for i in I_final]
        
        return aggregated_update, good_clients

    def aggregate(self, eligible, client_models):
        with torch.no_grad():
            client_models = list(client_models)

            # Get the client's updates
            server_model = parameters_to_vector(self.model.parameters())
            updates = torch.stack([
                parameters_to_vector(c.parameters()) - server_model
                for c in client_models
            ])
            n, d = updates.shape    # (eligible clients, numer of parameters)
            if self.b > d:
                raise ValueError(f"Subsampling parameter 'b'={self.b} is greater then the maximum number of parameter 'd'={d}. Must be equale or lower.")

            m = int(n * self.mal_perc)    # number of malicious client
            
            # call DnC aggegation function
            aggregated_update, good_clients = self._dnc_aggregation_rule(n, d, m, updates, client_models)

            # self.eta = 1.0 for FedAVG
            agg_model = server_model + self.lr * aggregated_update
            vector_to_parameters(agg_model, self.model.parameters())

            # Do the same for the buffers
            # Get model buffers
            model_buffers = dict(self.model.named_buffers())    # Includes running_mean, running_var, etc.
            good_buffers = [dict(c.named_buffers()) for c in good_clients]

            for key, buffer in model_buffers.items():
                if "num_batches_tracked" not in key:
                    stacked = torch.stack([cb[key].data for cb in good_buffers], dim=0)
                    buffer.data.copy_(stacked.mean(dim=0))

        return self.model


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGSignFlippingMaliciousClient
    
    def get_server_class(self):
        return FedAVGDnCDefenseServer