class T5Seq2Seq_hyperconfig:
    def __init__(self):
        super().__init__()
        self.lr = [1.0, 0.1, 0.01, 0.001]


    def process_config(self, config):
        return config

