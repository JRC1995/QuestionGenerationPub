import torch as T
import copy
from models.transformers import AutoTokenizer


class BinaryClassifier_collater:
    def __init__(self, PAD, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])
        self.PAD = PAD

    def pad(self, items, PAD, add_cls=False):
        if add_cls:
            items = [[self.cls] + item for item in items]
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def truncate(self, tokenized_obj):
        if len(tokenized_obj) > 510:
            tokenized_obj = tokenized_obj[0:510]

        tokenized_obj = [self.tokenizer.cls_token_id] + tokenized_obj + [self.tokenizer.sep_token_id]

        return tokenized_obj

    def slice_batch(self, batch, start_id, end_id):
        batch_ = {}
        for key in batch:
            batch_[key] = batch[key][start_id:end_id]
        return batch_

    def collate_fn(self, batch):
        sentences = [obj["sentence"] for obj in batch]
        tokenized_sentences = [self.truncate(self.tokenizer.encode(sentence, add_special_tokens=False)) for sentence in sentences]
        labels = [obj["class"] for obj in batch]

        batch_size = len(batch)

        input_vec, input_mask = self.pad(tokenized_sentences, PAD=self.PAD)

        batch = {}
        batch["sentences"] = sentences
        if self.config["DataParallel"]:
            batch["input_vec"] = T.tensor(input_vec).long()
            batch["labels"] = T.tensor(labels).float()
            batch["input_mask"] = T.tensor(input_mask).float()
        else:
            batch["input_vec"] = T.tensor(input_vec).long().to(self.config["device"])
            batch["labels"] = T.tensor(labels).float().to(self.config["device"])
            batch["input_mask"] = T.tensor(input_mask).float().to(self.config["device"])
        batch["batch_size"] = batch_size

        batches = [batch]

        return batches
