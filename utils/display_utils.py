import random

def step_display_fn(epoch, iter, item, args, config, extras):
    if args.model_type == "NLI":
        display_string = "Model: {}, Dataset: {}, Epoch {}, Step: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(
            config["model_name"],
            args.dataset,
            epoch,
            iter,
            item["metrics"]["loss"],
            item["metrics"]["accuracy"])
    elif args.model_type == "Seq2Seq" or args.model_type == "Seq2Seq_HVAE":
        display_string = "Model: {}, Dataset: {}, Epoch {}, Step: {}, Loss: {:.3f}".format(
            config["model_name"],
            args.dataset,
            epoch,
            iter,
            item["metrics"]["loss"])

    elif args.model_type == "BinaryClassifier":
        display_string = "Model: {}, Dataset: {}, Epoch {}, Step: {}, Loss: {:.3f}, Accuracy: {:.2f}".format(
            config["model_name"],
            args.dataset,
            epoch,
            iter,
            item["metrics"]["loss"],
            item["metrics"]["accuracy"])

    elif args.model_type == "HierarchicalLabeler":
        display_string = "Model: {}, Dataset: {}, Epoch {}, Step: {}, Loss: {:.3f}, Accuracy: {:.2f}".format(
            config["model_name"],
            args.dataset,
            epoch,
            iter,
            item["metrics"]["loss"],
            item["metrics"]["accuracy"])

    return display_string


def example_display_fn(epoch, iter, item, args, config, extras):
    if args.model_type == "NLI":
        idx2labels = extras["idx2labels"]
        item_len = len(item["display_items"]["predictions"])
        chosen_id = random.choice([id for id in range(item_len)])
        display_string = "Example:\nSequence1: {}\nSequence2: {}\nPrediction: {}\nGround Truth: {}\n".format(
            " ".join(item["display_items"]["sequences1"][chosen_id]),
            " ".join(item["display_items"]["sequences2"][chosen_id]),
            idx2labels[item["display_items"]["predictions"][chosen_id]],
            idx2labels[item["display_items"]["labels"][chosen_id]])
    elif args.model_type == "Seq2Seq" or args.model_type == "Seq2Seq_HVAE":
        item_len = len(item["display_items"]["predictions"])
        chosen_id = random.choice([id for id in range(item_len)])
        display_string = "Example:\nSource: {}\nTarget: {}\nPrediction: {}\n".format(
            item["display_items"]["source"][chosen_id],
            item["display_items"]["target"][chosen_id],
            item["display_items"]["predictions"][chosen_id])

    elif args.model_type == "BinaryClassifier":
        idx2labels = extras["idx2labels"]
        item_len = len(item["display_items"]["predictions"])
        chosen_id = random.choice([id for id in range(item_len)])

        prediction = item["display_items"]["predictions"][chosen_id]
        label = item["display_items"]["labels"][chosen_id]

        display_prediction = []
        display_label = []

        for id in range(len(prediction)):
            if prediction[id] == 1:
                display_prediction.append(idx2labels[id])
            else:
                display_prediction.append("not_"+idx2labels[id])

            if label[id] == 1:
                display_label.append(idx2labels[id])
            else:
                display_label.append("not_"+idx2labels[id])


        display_string = "Example:\nSentence: {}\nPrediction: {}\nGround Truth: {}\n".format(
            item["display_items"]["sentences"][chosen_id],
            ";".join(display_prediction),
            ";".join(display_label))

    elif args.model_type == "HierarchicalLabeler":
        item_len = len(item["display_items"]["predictions"])
        chosen_id = random.choice([id for id in range(item_len)])

        prediction = item["display_items"]["predictions"][chosen_id]
        label = item["display_items"]["labels"][chosen_id]

        display_prediction = [str(v) for v in prediction]
        display_label = [str(v) for v in label]

        display_string = "Example:\nSentence: {}\nPrediction: {}\nGround Truth: {}\n".format(
            item["display_items"]["paragraphs"][chosen_id],
            ";".join(display_prediction),
            ";".join(display_label))

    return display_string

def display(display_string, log_paths):
    with open(log_paths["log_path"], "a") as fp:
        fp.write(display_string)
    with open(log_paths["verbose_log_path"], "a") as fp:
        fp.write(display_string)
    print(display_string)

