import torch as T
import torch.nn as nn
from models.transformers import AutoTokenizer
from controllers.optimizer_controller import get_optimizer
import copy
import numpy as np


class BinaryClassifier_agent:
    def __init__(self, model, config, device, data=None):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if data is not None:
            self.label_weights = data["label_weights"]
        else:
            self.label_weights = None
        optimizer = get_optimizer(config)
        self.optimizer = optimizer(self.parameters,
                                   lr=config["lr"],
                                   weight_decay=config["weight_decay"])
        self.scheduler = None
        self.config = config
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.criterion = nn.BCELoss(reduction='none')
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])

    def loss_fn(self, logits, labels, label_weights=None):
        N, C = logits.size()
        loss = self.criterion(T.sigmoid(logits), labels)
        assert loss.size() == (N, C)

        if label_weights is not None:
            loss = (label_weights * loss).mean()
        else:
            loss = loss.mean()
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
        N, C = logits.size()
        labels = labels.view(N, C)

        if self.label_weights is None:
            label_weights = None
        else:
            label_weights = T.tensor(self.label_weights).float().view(1, C).to(labels.device)
            label_weights = T.where(labels == 1,
                                    label_weights,
                                    T.ones_like(label_weights).float().to(labels.device))

        loss = self.loss_fn(logits, labels, label_weights)

        predictions = np.where(T.sigmoid(logits).cpu().detach().numpy() >= 0.5,
                               1,
                               0).tolist()

        metrics = self.evaluate(batch_predictions=predictions,
                                batch_labels=labels.long().cpu().detach().numpy().tolist())

        if loss is not None:
            metrics["loss"] = loss.item()
        else:
            metrics["loss"] = 0.0

        item = {"display_items": {"sentences": batch["sentences"],
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

        idx2correct = {}
        idx2tp = {}
        idx2predictions = {}
        idx2golds = {}
        idx2total = {}

        for predictions, labels in zip(batch_predictions, batch_labels):
            i = 0

            for prediction, label in zip(predictions, labels):
                if prediction == label:
                    idx2correct[i] = idx2correct.get(i, 0) + 1
                    if label == 1:
                        idx2tp[i] = idx2tp.get(i, 0) + 1
                if prediction == 1:
                    idx2predictions[i] = idx2predictions.get(i, 0) + 1
                if label == 1:
                    idx2golds[i] = idx2golds.get(i, 0) + 1
                idx2total[i] = idx2total.get(i, 0) + 1
                i += 1

        per_label_accuracy = [idx2correct.get(i, 0) / idx2total.get(i, 0) for i in range(len(idx2total))]
        avg_accuracy = np.mean(per_label_accuracy) * 100

        metrics = {"idx2correct": idx2correct,
                   "idx2tp": idx2tp,
                   "idx2predictions": idx2predictions,
                   "idx2golds": idx2golds,
                   "idx2total": idx2total,
                   "accuracy": avg_accuracy}

        return metrics
