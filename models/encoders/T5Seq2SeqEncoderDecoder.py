import torch as T
import torch.nn as nn
from models.transformers.modeling_t5 import T5ForConditionalGeneration


class T5Seq2SeqEncoderDecoder(nn.Module):
    def __init__(self, config):

        super(T5Seq2SeqEncoderDecoder, self).__init__()

        self.embedding_path = config["embedding_path"]
        self.model = T5ForConditionalGeneration.from_pretrained(self.embedding_path, return_dict=True)
        self.config = config
        self.model.extra_config = config

    # %%

    def forward(self, src, src_mask, trg=None, trg_mask=None, generate=False):

        N, S = src.size()

        if not generate:
            outputs = self.model(input_ids=src,
                                 labels=trg,
                                 attention_mask=src_mask,
                                 decoder_attention_mask=trg_mask)

            logits = outputs.logits

            prediction = T.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        else:
            outputs = self.model.generate(input_ids=src,
                                          use_cache=False,
                                          do_sample=self.config["do_sample"],
                                          max_length=100,
                                          top_p=self.config["top_p"],
                                          num_beams=self.config["num_beams"],
                                          attention_mask=src_mask,
                                          num_return_sequences=self.config["num_returns"])

            prediction = outputs
            prediction = prediction.view(N, self.config["num_beams"] * self.config["num_returns"], -1)
            prediction = prediction.detach().cpu().numpy().tolist()

            logits = None

        return {"logits": logits, "prediction": prediction}
