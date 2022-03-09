import torch.nn as nn
from controllers.encoder_controller import encoder
from models.layers import Linear
from models.utils import gelu
import torch as T
import numpy as np
from models.transformers import ElectraModel


class HierarchicalLabeler_model(nn.Module):
    def __init__(self, attributes, config):
        super(HierarchicalLabeler_model, self).__init__()
        self.config = config
        self.encoder = ElectraModel.from_pretrained(config["embedding_path"], return_dict=True)

        self.hidden_size = attributes["hidden_size"]
        self.lstm_hidden_size = config["lstm_hidden_size"]

        self.compress = Linear(self.hidden_size, 150)

        self.rnn = nn.LSTM(input_size=self.lstm_hidden_size,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)

        self.layer1 = Linear(2*self.lstm_hidden_size, 2*self.lstm_hidden_size)
        self.layer2 = Linear(2*self.lstm_hidden_size, 1)

    # %%
    def forward(self, batch):
        # EMBEDDING BLOCK
        N, P, S = batch["input_vec"].size()

        input_vec = batch["input_vec"].view(N*P, S)
        input_mask = batch["input_mask"].view(N*P, S)

        outputs = self.encoder(input_ids=input_vec,
                               attention_mask=input_mask)
        encoded_vec = outputs.last_hidden_state
        assert encoded_vec.size() == (N*P, S, self.hidden_size)

        encoded_vec = encoded_vec[:, 0, :]
        encoded_vec = encoded_vec.view(N, P, self.hidden_size)

        encoded_vec = self.compress(encoded_vec)
        assert encoded_vec.size() == (N, P, self.lstm_hidden_size)

        input_mask = input_mask[:, 0]
        input_mask = input_mask.view(N, P)

        lengths = T.sum(input_mask, dim=1).int().view(N).cpu()
        packed_sequence = nn.utils.rnn.pack_padded_sequence(encoded_vec, lengths, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(packed_sequence)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        assert hidden_states.size() == (N, P, 2*self.lstm_hidden_size)

        logits = self.layer2(gelu(self.layer1(hidden_states)))

        return {"logits": logits}
