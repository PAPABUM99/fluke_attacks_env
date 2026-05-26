"""This module implement a :mod:`Mixin` class for malicious clients to be created."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from fluke import DDict
from fluke.client import Client, MaliciousClient
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader
from fluke.utils import get_loss

__all__ = [
    "MaliciousMixin", 
    "fedavg_dnc_def",
    "fedavg_gaussian",
    "fedavg_uniform",
    "fedavg_laplace",
    "fedavg_median_def",
    "fedavg_min_max",
    "fedavg_sign_flipping", 
    "fedavg_trimmed_mean_def", 
    "fedavg_multi_krum_def", 
    "fedala_attack",
    ]

class MaliciousMixin():
    """ 
    """
    def init_clients(
        self,
        clients_tr_data: list[FastDataLoader],
        clients_te_data: list[FastDataLoader],
        config: DDict,
    ) -> Sequence[Client]:
        """Creates the benign clients and malicious clients.

        Args:
            clients_tr_data (list[FastDataLoader]): List of training data loaders, one for
                each client.
            clients_te_data (list[FastDataLoader]): List of test data loaders, one for
                each client. The test data loaders can be ``None``.
            config (DDict): Configuration of the clients.

        Important:
            For more details about the configuration of the clients, see the
            :ref:`configuration <configuration>` page.

        See Also:
            :class:`fluke.client.Client`

        Returns:
            Sequence[Client]: List of initialized clients.
        """
        # alias
        protocol = self.cfg.protocol
        # clients percentage
        n_normal_clients = protocol.n_clients - int(protocol.n_clients * protocol.malicious_perc)
        n_malicious_clients = protocol.n_clients - n_normal_clients

        self._fix_opt_cfg(config.optimizer)
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer, scheduler_cfg=config.scheduler
        )
        loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        # normal clients
        clients = [
            self.get_client_class()(
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(loss),
                **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
            )
            for i in range(n_normal_clients)
        ]

        # malicious clients
        if n_malicious_clients > 0:
            mal_clients = [
                self.get_malicious_client_class()(
                    index=i,
                    train_set=clients_tr_data[i],
                    test_set=clients_te_data[i],
                    optimizer_cfg=optimizer_cfg,
                    loss_fn=deepcopy(loss),
                    malicious_hp=self.cfg.method.hyperparameters.malicious_clients,
                    **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
                )
                for i in range(n_malicious_clients)
            ]
            clients.extend(mal_clients)
        print(f"\n{clients[0]}\n{clients[-1]}\n")
        return clients

    def get_malicious_client_class(self):
        return MaliciousClient
