import torch as T
import copy
from models.transformers import AutoTokenizer


class Seq2Seq_collater:
    def __init__(self, PAD, config):
        self.PAD = PAD
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])
        self.cls = self.tokenizer.encode("<cls>", add_special_tokens=False)[0]
        self.sep = self.tokenizer.encode("<sep>", add_special_tokens=False)[0]
        self.hl_begin = self.tokenizer.encode("<hl>", add_special_tokens=False)[0]
        self.hl_end = self.tokenizer.encode("<\hl>", add_special_tokens=False)[0]

        assert len(self.tokenizer.encode("<cls>", add_special_tokens=False)) == 1
        assert len(self.tokenizer.encode("<sep>", add_special_tokens=False)) == 1
        assert len(self.tokenizer.encode("<hl>", add_special_tokens=False)) == 1
        assert len(self.tokenizer.encode("<\hl>", add_special_tokens=False)) == 1

    def pad(self, items, PAD, add_cls=True):
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
        if len(tokenized_obj) > 512:
            tokenized_obj = tokenized_obj[0:511] + [tokenized_obj[-1]]
            """
            if self.hl_begin in temp_tokenized_obj and self.hl_end in temp_tokenized_obj:
                tokenized_obj = temp_tokenized_obj
            else:
                hl_begin_id = tokenized_obj.index(self.hl_begin)
                component_tokenized_object = tokenized_obj[hl_begin_id:min(len(tokenized_obj)-1, hl_begin_id+511)] + [tokenized_obj[-1]]

                if "type" in self.config["dataset"]:
                    rem = 511 - len(component_tokenized_object)
                    tokenized_obj = [tokenized_obj[0]] + tokenized_obj[max(1, hl_begin_id-rem):hl_begin_id] + component_tokenized_object
                else:
                    rem = 512 - len(component_tokenized_object)
                    tokenized_obj = [tokenized_obj[max(0, hl_begin_id-rem):hl_begin_id] + component_tokenized_object
            """

        return tokenized_obj

    def slice_batch(self, batch, start_id, end_id):
        batch_ = {}
        for key in batch:
            if key != "batch_size":
                batch_[key] = batch[key][start_id:end_id]
            else:
                batch_[key] = int(end_id - start_id)

        assert batch_["src_vec"].size(0) > 0
        assert batch_["trg_vec"].size(0) > 0
        assert batch_["label"].size(0) > 0
        assert batch_["src_mask"].size(0) > 0
        assert batch_["trg_mask"].size(0) > 0

        return batch_

    def collate_fn(self, batch):
        src = [obj["document"] for obj in batch]
        tokenized_src = [self.truncate(self.tokenizer.encode(obj["document"])) for obj in batch]
        batch_question = [obj["question"] for obj in batch]
        batch_questions = [obj["questions"] for obj in batch]

        trg = [obj["question"] for obj in batch]
        tokenized_trg = [self.tokenizer.encode(obj["question"]) for obj in batch]

        batch_size = len(tokenized_src)

        src_vec, src_mask = self.pad(tokenized_src, PAD=self.PAD)
        if self.config["vae"]:
            add_cls = True
        else:
            add_cls = False
        trg_vec, trg_mask = self.pad(copy.deepcopy(tokenized_trg), PAD=self.PAD, add_cls=add_cls)
        label, _ = self.pad(tokenized_trg, PAD=-100, add_cls=False)

        batch = {}
        if self.config["DataParallel"]:
            batch["src_vec"] = T.tensor(src_vec).long()
            batch["trg_vec"] = T.tensor(trg_vec).long()
            batch["label"] = T.tensor(label).long()
            batch["src_mask"] = T.tensor(src_mask).float()
            batch["trg_mask"] = T.tensor(trg_mask).float()
        else:
            batch["src_vec"] = T.tensor(src_vec).long().to(self.config["device"])
            batch["trg_vec"] = T.tensor(trg_vec).long().to(self.config["device"])
            batch["label"] = T.tensor(label).long().to(self.config["device"])
            batch["src_mask"] = T.tensor(src_mask).float().to(self.config["device"])
            batch["trg_mask"] = T.tensor(trg_mask).float().to(self.config["device"])

        batch["src"] = src
        batch["trg"] = trg

        batch["batch_size"] = batch_size
        batch["questions"] = batch_questions
        batch["question"] = batch_question

        if (batch["src_vec"].size(1) <= 300 and batch["trg_vec"].size(1) <= 100) or (batch_size == 1):
            assert batch["src_vec"].size(0) > 0
            assert batch["trg_vec"].size(0) > 0
            assert batch["label"].size(0) > 0
            assert batch["src_mask"].size(0) > 0
            assert batch["trg_mask"].size(0) > 0
            batches = [batch]
        else:
            batches = [self.slice_batch(batch, 0, batch_size // 2),
                       self.slice_batch(batch, batch_size // 2, batch_size)]

        return batches
