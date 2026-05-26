import torch

from fluke.algorithms.attacks import MaliciousMixin
from fluke.algorithms.fedavg import FedAVG
from fluke.client import MaliciousClient
from fluke.utils import clear_cuda_cache

__all__ = ["FedAVGAttack", "FedAVGSignFlippingMaliciousClient"]


# For use SignFlipping attack it's necessary to write in [ALG_CONFIG] under 'malicious_clients' sigma: float 
class FedAVGSignFlippingMaliciousClient(MaliciousClient):
    def fit(self, override_local_epochs = 0):
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        # save a copy of the model params before training 
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

        # update the trained model with the negated params
        with torch.no_grad():
            for key, param in self.model.named_parameters():
                diff = param.data - pre_train_params[key]
                param.data.copy_(pre_train_params[key] - self.malicious_hp.sigma * diff)

        train_batches = epochs * len(self.train_set)
        if train_batches > 0:
            running_loss /= train_batches

        self.model.cpu()
        clear_cuda_cache()
        return running_loss


class FedAVGAttack(MaliciousMixin, FedAVG):
    def get_malicious_client_class(self):
        return FedAVGSignFlippingMaliciousClient