import torch

from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.attacks.fedavg_gaussian import FedAVGGaussianMaliciousClient
from fluke.algorithms.attacks.fedavg_laplace import FedAVGLaplaceMaliciousClient
from fluke.algorithms.attacks.fedavg_min_max import FedAVGMinMaxMaliciousClient
from fluke.algorithms.attacks.fedavg_sign_flipping import FedAVGSignFlippingMaliciousClient
from fluke.algorithms.attacks.fedavg_uniform import FedAVGUniformMaliciousClient
from fluke.algorithms.fedavg import FedAVG
from fluke.server import Server

__all__ = ["FedAVGAttack", "FedAVGMedianDefenceServer"]


class FedAVGMedianDefenceServer(Server):
    """Aggregate the models using the median of all parameter for each client's model"""
    def aggregate(self, eligible, client_models):
        client_models = list(client_models) # client's model list
        n = len(client_models)  # total client models
        
        # Get model parameters and buffers
        model_params = dict(self.model.named_parameters())
        model_buffers = dict(self.model.named_buffers())  # Includes running_mean, running_var, etc.

        # Get clients parameters and buffers
        client_params = [dict(m.named_parameters()) for m in client_models]
        client_buffer = [dict(m.named_buffers()) for m in client_models]

        def coordinate_median(stacked, n):
            sorted_vals = torch.sort(stacked, dim=0).values
            mid = n // 2
            if n % 2 == 1:
                return sorted_vals[mid]
            else:
                return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])

        # Compute median over all parameters for each client, per layer
        for key in model_params.keys():
            agg_params = torch.stack([cp[key].data for cp in client_params])
            model_params[key].data.copy_(coordinate_median(agg_params, n))
        
        for key in model_buffers.keys():
            if "num_batches_tracked" in key:
                continue
            else:
                agg_buffer = torch.stack([cb[key].data for cb in client_buffer])
                model_buffers[key].data.copy_(coordinate_median(agg_buffer, n))
        
        return self.model


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGMinMaxMaliciousClient

    def get_server_class(self):
        return FedAVGMedianDefenceServer