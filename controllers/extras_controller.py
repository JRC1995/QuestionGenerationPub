
def extras_fn(data, args):
    if args.model_type == "NLI" or args.model_type == "BinaryClassifier":
        extras = {"idx2labels": data["idx2labels"]}
    else:
        extras = None

    return extras