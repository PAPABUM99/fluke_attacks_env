""" Implementation of the (Min-Max) Minimize maximum distance attack
    
    References:
    Shejwalkar, Virat & Houmansadr, Amir. (2021). 
    Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. 
    10.14722/ndss.2021.24498. 
    URL: https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/
"""

import torch

from fluke import DDict
from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.fedavg import FedAVG
from fluke.client import MaliciousClient
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader
from fluke.utils import clear_cuda_cache


__all__ = ["FedAVGAttack", "FedAVGMinMaxMaliciousClient"]

# Per farlo funzionare nel [ALG_CONFIG] in malicious_hp imposta 
# 'eligible_perc: float' e 'malicious_perc: float'
class FedAVGMinMaxMaliciousClient(MaliciousClient):
    def _benign_estimation(
            self, 
            update: torch.Tensor, 
            mal_perc: float, 
            eligible: float,
        ):
        k = int((eligible * 100) * (1 - mal_perc))   # benign clients estimation
        noise = update.std().item()
        estimations = [update]

        for _ in range(k - 1):
            estimate = update + torch.randn_like(update) * noise
            estimations.append(estimate)
        
        return torch.stack(estimations) 

    def _perturbation_sign(self, avg_updates: torch.Tensor) -> torch.Tensor:
        return -torch.sign(avg_updates)

    def _oracle_minmax(
            self, 
            benign_updates: torch.Tensor, 
            gamma: float, 
            perturbation: torch.Tensor,
        ) -> bool:
        avg_update = benign_updates.mean(dim=0)
        malicious_update = avg_update + gamma * perturbation

        # max distance between malicious and benign
        max_mal_dist = torch.stack([
            torch.linalg.norm(malicious_update - bu)
            for bu in benign_updates
        ]).max()

        # max distance among benign updates
        max_benign_dist = torch.stack([
            torch.linalg.norm(bu1 - bu2)
            for i, bu1 in enumerate(benign_updates) 
            for j, bu2 in enumerate(benign_updates)
            if i != j
        ]).max()

        return max_mal_dist <= max_benign_dist

    def _optimized_gamma(
            self,
            delta_b: torch.Tensor,
            perturbation: torch.Tensor,
            gamma_init: float = 10.0,
            tau: float = 1e-4,
    ) -> float:
        gamma = gamma_init
        step = gamma_init / 2.0
        gamma_succ = 0.0

        while step > tau:
            if self._oracle_minmax(delta_b, gamma, perturbation) == True:
                gamma_succ = gamma
                gamma += step / 2.0
            else:
                gamma -= step / 2.0
            step /= 2.0
        
        return gamma_succ

    def fit(self, override_local_epochs = 0):
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        # save model's params dict before training
        pre_train_params = {key: param.data.clone() for key, param in self.model.named_parameters()}

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        # save local update
        benign_update = torch.cat([
            (param.data - pre_train_params[key]).ravel() 
            for key, param in self.model.named_parameters()
        ])

        # estimate k benign updates to simulate the other client's updates
        # we add gaussian noise for this purpose
        estimated_benign_updates = self._benign_estimation(
            update=benign_update, 
            mal_perc=self.malicious_hp.malicious_perc,
            eligible=self.malicious_hp.eligible_perc
        )

        # Min-Max Attack
        # calculate perturbation
        mu = estimated_benign_updates.mean(dim=0)
        perturbation = self._perturbation_sign(mu)

        # gamma optimization
        gamma = self._optimized_gamma(
            delta_b=estimated_benign_updates, 
            perturbation=perturbation
        )

        # calculate malicious update
        malicious_update = benign_update + gamma * perturbation
        
        # copy malicious update
        offset = 0
        with torch.no_grad():
           for key, param in self.model.named_parameters():
                numel = param.numel()
                param.data.copy_(
                    malicious_update[offset:offset + numel].view_as(param)
                )
                offset += numel

        train_batches = epochs * len(self.train_set)
        if train_batches > 0:
            running_loss /= train_batches
        
        self.model.cpu()
        clear_cuda_cache()
        return running_loss
    
class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGMinMaxMaliciousClient