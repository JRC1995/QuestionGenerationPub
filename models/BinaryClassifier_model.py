import torch.nn as nn
from controllers.encoder_controller import encoder
from models.layers import Linear
from models.utils import gelu
import torch as T
import numpy as np
from models.transformers import ElectraModel


class BinaryClassifier_model(nn.Module):
    def __init__(self, attributes, config):
        super(BinaryClassifier_model, self).__init__()
        self.config = config
        self.encoder = ElectraModel.from_pretrained(config["embedding_path"], return_dict=True)
        self.hidden_size = attributes["hidden_size"]
        self.classes_num = attributes["classes_num"]
        self.layer1 = Linear(self.hidden_size, self.hidden_size)
        self.layer2 = Linear(self.hidden_size, self.classes_num)

    # %%
    def forward(self, batch):
        # EMBEDDING BLOCK
        N, S = batch["input_vec"].size()
        outputs = self.encoder(input_ids=batch["input_vec"],
                               attention_mask=batch["input_mask"])
        encoded_vec = outputs.last_hidden_state
        assert encoded_vec.size() == (N, S, self.hidden_size)

        encoded_vec = encoded_vec[:, 0, :]
        logits = self.layer2(gelu(self.layer1(encoded_vec)))

        return {"logits": logits}
