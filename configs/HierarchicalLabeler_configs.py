from os import fspath
from pathlib import Path


class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 5
        self.batch_size = 64
        self.train_batch_size = 2
        self.dev_batch_size = 2
        self.DataParallel = False
        self.weight_decay = 1e-2
        self.num_workers = 0
        self.lr = 2e-5
        self.epochs = 30
        self.early_stop_patience = 2
        self.optimizer = "RAdam"
        self.warm_up_steps = 2000


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.lstm_hidden_size = 150
        self.embedding_path = fspath(Path("embeddings/ELECTRA_base"))

class ELECTRAHierarchicalLabeler_config(base_config):
    def __init__(self):
        super().__init__()
        self.model_name = "(ELECTRA Hierarchical Labeler)"


