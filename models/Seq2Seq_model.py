import torch.nn as nn
from controllers.encoder_controller import encoder


class Seq2Seq_model(nn.Module):
    def __init__(self, attributes, config):

        super(Seq2Seq_model, self).__init__()

        self.config = config
        self.encoder_decoder = encoder(config)


    # %%
    def forward(self, batch, generate=False):

        src = batch["src_vec"]
        src_mask = batch["src_mask"]
        if "trg_vec" in batch:
            trg = batch["trg_vec"]
            trg_mask = batch["trg_mask"]
        else:
            trg = None
            trg_mask = None


        # EMBEDDING BLOCK
        sequence_dict = self.encoder_decoder(src, src_mask, trg, trg_mask, generate)

        return sequence_dict
