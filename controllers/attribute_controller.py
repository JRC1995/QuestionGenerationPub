
def prepare_attributes(args, data=None):
    if args.model_type == "BinaryClassifier":
        if args.model == "ELECTRABinaryClassifier":
            attributes = {"classes_num": 1,
                          "hidden_size": 768}
        elif args.model == "ELECTRAMultiLabelClassifier":
            attributes = {"classes_num": data["classes_num"],
                          "hidden_size": 768}
    elif args.model_type == "HierarchicalLabeler":
        attributes = {"hidden_size": 768}
    else:
        attributes = None

    return attributes