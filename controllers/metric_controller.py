def compute_F1(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def metric_fn(metrics, args, data):

    if args.model_type == "BinaryClassifier":

        labels2idx = data["labels2idx"]
        idx2labels = {v: k for k, v in labels2idx.items()}

        classes_num = len(metrics[0]["idx2total"])

        idx2correct = {}
        idx2tp = {}
        idx2predictions = {}
        idx2golds = {}
        idx2total = {}
        idx2accuracy = {}
        composed_metric = {}

        for id in range(classes_num):
            idx2correct[id] = sum([metric["idx2correct"].get(id, 0) for metric in metrics])
            idx2total[id] = sum([metric["idx2total"].get(id, 0) for metric in metrics])
            composed_metric[idx2labels[id] + "_accuracy"] = (idx2correct[id] / idx2total[id]) * 100

            idx2tp[id] = sum([metric["idx2tp"].get(id, 0) for metric in metrics])
            idx2predictions[id] = sum([metric["idx2predictions"].get(id, 0) for metric in metrics])
            idx2golds[id] = sum([metric["idx2golds"].get(id, 0) for metric in metrics])

            composed_metric[idx2labels[id] + "_precision"] \
                = idx2tp[id] / idx2predictions[id] if idx2predictions[id] > 0 else 0
            composed_metric[idx2labels[id] + "_recall"] = idx2tp[id] / idx2golds[id] if idx2golds[id] > 0 else 0

            composed_metric[idx2labels[id] + "_F1"] = compute_F1(prec=composed_metric[idx2labels[id] + "_precision"],
                                                                 rec=composed_metric[idx2labels[id] + "_recall"])

        loss = sum([metric["loss"] * metric["idx2total"][0] for metric in metrics]) / idx2total[0]
        composed_metric["loss"] = loss

    elif args.model_type == "HierarchicalLabeler":

        composed_metric = {}

        correct = sum([metric["correct"] for metric in metrics])
        total = sum([metric["total"] for metric in metrics])
        composed_metric["accuracy"] = (correct / total) * 100

        tp = sum([metric["tp"] for metric in metrics])
        predictions = sum([metric["predictions"] for metric in metrics])
        golds = sum([metric["golds"] for metric in metrics])

        composed_metric["precision"] = tp / predictions if predictions > 0 else 0
        composed_metric["recall"] = tp / golds if golds > 0 else 0

        composed_metric["F1"] = compute_F1(prec=composed_metric["precision"],
                                           rec=composed_metric["recall"])

        loss = sum([metric["loss"] * metric["total"] for metric in metrics]) / total
        composed_metric["loss"] = loss


    elif args.model_type == "Seq2Seq":
        composed_metric = {}
        total_data = sum([metric["total_data"] for metric in metrics])
        loss = sum([metric["loss"] * metric["total_data"] for metric in metrics]) / total_data
        composed_metric["loss"] = loss

    return composed_metric


def compose_dev_metric(metrics, args):
    total_loss = 0
    n = len(metrics)
    for key in metrics:
        total_loss += metrics[key]["loss"]
    return -total_loss / n
