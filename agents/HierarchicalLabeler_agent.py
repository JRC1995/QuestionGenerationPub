import torch as T
import torch.nn as nn
from models.transformers import AutoTokenizer
from controllers.optimizer_controller import get_optimizer
import copy
import numpy as np


class HierarchicalLabeler_agent:
    def __init__(self, model, config, device, data=None):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        #self.label_weights = data["label_weights"]
        optimizer = get_optimizer(config)
        self.optimizer = optimizer(self.parameters,
                                   lr=config["lr"],
                                   weight_decay=config["weight_decay"])
        self.scheduler = None
        self.config = config
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.criterion = nn.BCELoss(reduction='mean')
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])

    def loss_fn(self, logits, labels):
        loss = self.criterion(T.sigmoid(logits), labels)
        return loss

    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        output_dict = self.model(batch)
        logits = output_dict["logits"]

        labels = batch["labels"].to(logits.device)

        # print(labels)
        N, P, _ = logits.size()
        logits = logits.view(N, P)
        labels = labels.view(N, P)

        loss = self.loss_fn(logits, labels)

        predictions = np.where(T.sigmoid(logits).cpu().detach().numpy() >= 0.5,
                               1,
                               0).tolist()

        metrics = self.evaluate(batch_predictions=predictions,
                                batch_labels=labels.long().cpu().detach().numpy().tolist())

        if loss is not None:
            metrics["loss"] = loss.item()
        else:
            metrics["loss"] = 0.0

        item = {"display_items": {"paragraphs": batch["paragraphs"],
                                  "labels": labels.long().cpu().detach().numpy().tolist(),
                                  "predictions": predictions},
                "loss": loss,
                "metrics": metrics,
                "stats_metrics": metrics}

        return item

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(self.parameters, self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evaluate(self, batch_predictions, batch_labels):

        correct = 0
        tp = 0
        positive_predictions = 0
        golds = 0
        total = 0

        for predictions, labels in zip(batch_predictions, batch_labels):
            for prediction, label in zip(predictions, labels):
                if label != -1:
                    if prediction == label:
                        correct += 1
                        if label == 1:
                            tp += 1
                    if prediction == 1:
                        positive_predictions += 1
                    if label == 1:
                        golds += 1
                    total += 1


        accuracy = (correct/ total) * 100

        metrics = {"correct": correct,
                   "tp": tp,
                   "predictions": positive_predictions,
                   "golds": golds,
                   "total": total,
                   "accuracy": accuracy}

        return metrics
