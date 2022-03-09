import torch as T
import torch.nn as nn
from models.transformers import AutoTokenizer
from controllers.optimizer_controller import get_optimizer
import copy

class Seq2Seq_agent:
    def __init__(self, model, config, device, data=None):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optimizer(config)
        self.optimizer = optimizer(self.parameters,
                                   lr=config["lr"],
                                   weight_decay=config["weight_decay"])
        lr_lambda = lambda global_step: min(1., (global_step / config["warm_up_steps"]) ** 2)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.config = config
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])

    def loss_fn(self, logits, labels, kd=None):
        loss = self.criterion(logits, labels)
        if kd is not None:
            loss = loss + kd
        return loss

    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()


        output_dict = self.model(batch, generate = self.config["generate"])
        logits = output_dict["logits"]
        predictions = output_dict["prediction"]
        if "kd" in output_dict:
            kd = output_dict["kd"]
        else:
            kd = None

        if not self.config["generate"]:
            predictions = [self.tokenizer.decode(prediction) for prediction in predictions]
        else:
            predictions_ = []
            for beam_prediction in predictions:
                beam_prediction = [self.tokenizer.decode(prediction) for prediction in beam_prediction]
                predictions_.append(beam_prediction)
            predictions = predictions_

        if logits is not None:
            labels = batch["label"].to(logits.device)

            #print(labels)

            N = logits.size(0)
            S = logits.size(1)

            logits = logits.view(N*S, -1)
            labels = labels.view(-1)

            loss = self.loss_fn(logits, labels, kd)

        else:
            loss = None

        metrics = {}
        metrics["total_data"] = batch["batch_size"]
        if loss is not None:
            metrics["loss"] = loss.item()
        else:
            metrics["loss"] = 0.0

        item = {"display_items": {"source": batch["src"],
                                   "target": batch["trg"],
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
        self.scheduler.step()

