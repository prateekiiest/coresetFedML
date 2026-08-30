"""
CoreSet-PFedBayes server loop (paper Algorithm 1, "Cloud server executes").

    for t in 0..T-1:
        broadcast z^t
        evaluate
        S^t = random subset of clients
        for i in S^t (parallel):  v^{t+1}_i = ClientUpdate(i, w_i, z^t)
        z^{t+1} = (1-beta) z^t + (beta/|S^t|) sum_i v^{t+1}_i
"""

import numpy as np
from tqdm import trange

from src.clientModels.clientBaseClass import Client
from src.clientModels.clientModelClass import ClientModelClass
from src.serverModels.serverBaseClass import Server
from utils.model_utils import read_data, read_user_data


class pFedBayes(Server):
    def __init__(self, global_model, template_model, cfg):
        super().__init__(global_model, cfg)
        self.template = template_model
        self.rng = np.random.default_rng(cfg["seed"])

        users, _, train_data, test_data = read_data(cfg["dataset"])
        n_users = min(cfg["num_users"], len(users))
        for i in range(n_users):
            uid, train, test = read_user_data(i, (users, _, train_data, test_data),
                                              cfg["dataset"], cfg["device"])
            base = Client(uid, train, test, cfg["output_dim"], cfg["batch_size"],
                          cfg["device"])
            self.users.append(ClientModelClass(base, template_model, cfg))
        print(f"[server] {len(self.users)} clients | method={cfg['method']} "
              f"| coreset_frac={cfg['coreset_frac']}")

    def train(self):
        for t in trange(self.cfg["num_glob_iters"], desc="rounds"):
            self.broadcast()
            self.evaluate(t)
            selected = self.select(self.rng)
            for u in selected:
                u.local_train()
            self.aggregate(selected)
        self.evaluate(self.cfg["num_glob_iters"])
        return self.save_results(self.cfg["tag"])
