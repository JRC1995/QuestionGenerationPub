from os import fspath
from pathlib import Path


class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 5
        self.batch_size = 128
        self.train_batch_size = 8
        self.dev_batch_size = 8
        self.DataParallel = False
        self.weight_decay = 0
        self.num_workers = 0
        self.lr = 0.01
        self.epochs = 20
        self.early_stop_patience = 2
        self.optimizer = "SM3"
        self.warm_up_steps = 2000



class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.num_beams = 1
        self.vae = False
        self.num_returns = 1
        self.top_p = 1.0
        self.do_sample = False
        self.embedding_path = fspath(Path("embeddings/T5_base_qg"))
        self.encoder = "T5Seq2SeqEncoderDecoder"
        self.generate = False
        self.type_driven = False



class T5Seq2Seq_config(base_config):
    def __init__(self):
        super().__init__()
        self.model_name = "(T5 Seq2Seq)"

class T5Seq2SeqVAE_config(base_config):
    def __init__(self):
        super().__init__()
        self.embedding_path = fspath(Path("embeddings/T5_base_qg"))
        self.encoder = "T5VAESeq2SeqEncoderDecoder"
        self.model_name = "(T5 Seq2Seq VAE)"
        self.vae = True
