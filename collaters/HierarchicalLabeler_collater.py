import torch as T
import copy
from models.transformers import AutoTokenizer


class HierarchicalLabeler_collater:
    def __init__(self, PAD, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])
        self.PAD = PAD

    def hierarchical_pad(self, paragraphs, PAD):

        all_sentence_lens = []
        all_paragraph_lens = []

        for paragraph in paragraphs:
            all_paragraph_lens.append(len(paragraph))
            for sentence in paragraph:
                all_sentence_lens.append(len(sentence))

        max_paragraph_len = max(all_paragraph_lens)
        max_sentence_len = max(all_sentence_lens)

        padded_paragraphs = []
        input_mask = []

        for paragraph in paragraphs:
            padded_paragraph = []
            paragraph_mask = []

            for sentence in paragraph:
                padded_sentence = sentence
                sentence_mask = [1] * len(sentence)
                while len(padded_sentence) < max_sentence_len:
                    padded_sentence.append(PAD)
                    sentence_mask.append(0)
                padded_paragraph.append(padded_sentence)
                paragraph_mask.append(sentence_mask)

            while len(padded_paragraph) < max_paragraph_len:
                padded_paragraph.append([PAD] * max_sentence_len)
                paragraph_mask.append([0]*max_sentence_len)

            padded_paragraphs.append(padded_paragraph)
            input_mask.append(paragraph_mask)

        return padded_paragraphs, input_mask

    def pad(self, items, PAD):
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
        batch_size = len(batch)
        paragraphs = [obj["paragraph"] for obj in batch]
        tokenized_paragraphs = []
        for paragraph in paragraphs:
            tokenized_paragraph = [self.truncate(self.tokenizer.encode(sentence, add_special_tokens=False)) for sentence
                                   in paragraph]
            tokenized_paragraphs.append(tokenized_paragraph)

        labels = [obj["label"] for obj in batch]

        input_vec, input_mask = self.hierarchical_pad(tokenized_paragraphs, PAD=self.PAD)
        labels, _ = self.pad(labels, PAD=-1)

        batch = {}
        batch["paragraphs"] = [" -|- ".join(paragraph) for paragraph in paragraphs]
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
