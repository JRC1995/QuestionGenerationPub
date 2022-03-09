import random
import zlib
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch.nn as nn
from models.transformers import AutoTokenizer
from nltk import sent_tokenize
from collaters import *
from configs.configLoader import load_config
from controllers.attribute_controller import prepare_attributes
from controllers.extras_controller import extras_fn
from controllers.metric_controller import metric_fn, compose_dev_metric
from argparser import get_args
from trainers import Trainer
from utils.checkpoint_utils import load_temp_checkpoint, load_infer_checkpoint, save_infer_checkpoint, \
    save_temp_checkpoint
from utils.data_utils import load_data, load_dataloaders, Dataset
from utils.display_utils import example_display_fn, step_display_fn, display
from utils.param_utils import param_display_fn, param_count
from utils.path_utils import load_paths
from models import *
from agents import *
from torch.utils.data import DataLoader
from collaters.HierarchicalLabeler_collater import HierarchicalLabeler_collater
from collaters.BinaryClassifier_collater import BinaryClassifier_collater
import jsonlines
from os import fspath
from tqdm import tqdm

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


def truncate(self, tokenized_obj):
    if len(tokenized_obj) > 510:
        tokenized_obj = tokenized_obj[0:510]

    tokenized_obj = [self.tokenizer.cls_token_id] + tokenized_obj + [self.tokenizer.sep_token_id]

    return tokenized_obj

def run(args, config, time=0):
    device = T.device(args.device)
    config["device"] = device

    SEED = "{}_{}_{}_{}".format(args.dataset, args.model, args.model_type, time)
    SEED = zlib.adler32(str.encode(SEED))
    display_string = "\n\nSEED: {}\n\n".format(SEED)
    display_string += "Parsed Arguments: {}\n\n".format(args)

    T.manual_seed(SEED)
    random.seed(SEED)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    display_string += "Configs:\n"
    for k, v in config.items():
        display_string += "{}: {}\n".format(k, v)
    display_string += "\n"

    if "sentence" in args.dataset:
        args_ = deepcopy(args)
        args_.model_type = "HierarchicalLabeler"
        args_.dataset = "SQuADDu_qw_hierarchical_labeling" if "SQuADDu" in args.dataset else "NewsQA_qw_hierarchical_labeling"
        args_.model = "ELECTRAHierarchicalLabeler"
        paths, checkpoint_paths, metadata = load_paths(args_, time)
        attributes = prepare_attributes(args_)
        config_ = load_config(args_)
        config_["device"] = device
        qw_model = HierarchicalLabeler_model(attributes=attributes,
                                             config=config_)
        qw_model = qw_model.to(device)
        if config_["DataParallel"]:
            qw_model = nn.DataParallel(qw_model)

        qw_agent = eval("{}_agent".format(args_.model_type))
        qw_agent = qw_agent(model=qw_model,
                            config=config_,
                            device=device)

        qw_agent = load_infer_checkpoint(qw_agent, checkpoint_paths, paths)
        qw_tokenizer = AutoTokenizer.from_pretrained(config_["embedding_path"])
        hcollater = HierarchicalLabeler_collater(PAD=qw_tokenizer.pad_token_id,
                                                 config=config_)

    if "type_driven" in args.dataset:
        args_ = deepcopy(args)
        args_.model_type = "BinaryClassifier"
        args_.dataset = "SQuADDu_qt" if "SQuADDu" in args.dataset else "NewsQA_qt"
        args_.model = "ELECTRAMultiLabelClassifier"
        paths, checkpoint_paths, metadata = load_paths(args_, time)
        data = load_data(paths, metadata, args_)
        idx2qtlabels = data["idx2labels"]
        assert idx2qtlabels is not None
        attributes = prepare_attributes(args_, data)
        config_ = load_config(args_)
        config_["device"] = device
        qt_model = BinaryClassifier_model(attributes=attributes,
                                          config=config_)
        qt_model = qt_model.to(device)
        if config_["DataParallel"]:
            qt_model = nn.DataParallel(qt_model)

        qt_agent = eval("{}_agent".format(args_.model_type))
        qt_agent = qt_agent(model=qt_model,
                            config=config_,
                            device=device)

        qt_agent = load_infer_checkpoint(qt_agent, checkpoint_paths, paths)
        qt_tokenizer = AutoTokenizer.from_pretrained(config_["embedding_path"])
        bcollater = BinaryClassifier_collater(PAD=qt_tokenizer.pad_token_id,
                                              config=config_)

    paths, checkpoint_paths, metadata = load_paths(args, time)
    # data = load_data(paths, metadata, args)
    attributes = prepare_attributes(args)
    model = eval("{}_model".format(args.model_type))
    model = model(attributes=attributes,
                  config=config)
    qg_model = model.to(device)

    if config["DataParallel"]:
        qg_model = nn.DataParallel(qg_model)

    qg_agent = eval("{}_agent".format(args.model_type))
    qg_agent = qg_agent(model=qg_model,
                        config=config,
                        device=device)

    qg_agent = load_infer_checkpoint(qg_agent, checkpoint_paths, paths)
    qg_tokenizer = AutoTokenizer.from_pretrained(config["embedding_path"])
    sep_token_id = qg_tokenizer.encode("<sep>", add_special_tokens=False)[0]

    args_ = deepcopy(args)
    args_.dataset = "SQuADDu_one2many_QG"
    paths_, _, metadata_ = load_paths(args_, time)
    data = load_data(paths_, metadata_, args_)

    if args.display_params:
        display_string += param_display_fn(qg_model)

    total_parameters = param_count(qg_model)
    display_string += "Total Parameters: {}\n\n".format(total_parameters)

    print(display_string)

    predict_dict = {}
    predict_list = []

    for key in data["test"]:
        test_samples = data["test"][key].samples
        for id in tqdm(test_samples):
            document = test_samples[id]["document"]
            if document not in predict_dict:
                if "sentence" not in args.dataset:
                    documents = [document]
                    qw_predictions = [1]
                else:
                    documents = sent_tokenize(document)
                    batch = [{"paragraph": documents, "label": [0] * len(documents)}]
                    batch = hcollater.collate_fn(batch)[0]
                    logits = qw_agent.model(batch)["logits"].view(len(documents))
                    qw_predictions = np.where(T.sigmoid(logits).cpu().detach().numpy() >= 0.5,
                                              1,
                                              0).tolist()

                predictions = []
                for doc_id, document_ in enumerate(documents):

                    if qw_predictions[doc_id] == 1:

                        question_types = []

                        if "type_driven" in args.dataset:
                            print("document_: ", document_)
                            batch = [{"sentence": document_,
                                      "class": [0] * 10}]
                            batch = bcollater.collate_fn(batch)[0]
                            logits = qt_agent.model(batch)["logits"].view(-1)
                            qt_predictions = np.where(T.sigmoid(logits).cpu().detach().numpy() >= 0.5,
                                                      1,
                                                      0).tolist()
                            print(qt_predictions)

                            max_id = np.argmax(logits.cpu().detach().numpy())

                            if max(qt_predictions) == 0:
                                question_types = [idx2qtlabels[max_id]]
                            else:
                                question_types = []
                                for qt_id, val in enumerate(qt_predictions):
                                    if val == 1:
                                        question_types.append(idx2qtlabels[qt_id])

                        if not question_types:
                            question_types = [None]

                        if "sentence" in args.dataset:
                            if "NewsQA" in args.dataset:
                                copy_documents = copy.deepcopy(documents)
                                as_id = copy_documents.index(document_)
                                if as_id > 2:
                                    selected_sentences = [copy_documents[as_id - 2], copy_documents[as_id - 1]] \
                                                         + copy_documents[as_id:]
                                else:
                                    selected_sentences = copy_documents
                                partial_document = " ".join(selected_sentences)
                                document_ = partial_document.replace(document_, " <hl> " + document_ + " </hl> ")
                            else:
                                document_ = document.replace(document_, " <hl> " + document_ + " </hl> ")

                        for question_type in question_types:

                            if question_type is not None:
                                document_with_type = "<" + question_type + "> " + document_
                                tokenized_document = qg_tokenizer.encode(document_with_type)
                                print(document_with_type)
                            else:
                                tokenized_document = qg_tokenizer.encode(document_)
                                print(document_)

                            tokenized_document = [qg_tokenizer.encode("<cls>", add_special_tokens=False)[
                                                      0]] + tokenized_document
                            S = len(tokenized_document)
                            batch = {"src_vec": T.tensor(tokenized_document).long().to(device).view(1, S),
                                     "src_mask": T.ones(1, S).float().to(device)}

                            config["do_sample"] = False
                            config["top_p"] = 1.0
                            config["num_returns"] = 1
                            predictions_ = qg_agent.model(batch, generate=True)["prediction"][0]

                            print(qg_tokenizer.decode(predictions_[0]))

                            if "one2one" in args.dataset:
                                config["do_sample"] = True
                                config["top_p"] = 0.9
                                if "sentence" in args.dataset and "type" not in args.dataset:
                                    config["num_returns"] = 9
                                elif "type" in args.dataset:
                                    config["num_returns"] = 4
                                else:
                                    config["num_returns"] = 19
                                predictions_ += qg_agent.model(batch, generate=True)["prediction"][0]

                            predictions = predictions + predictions_

                decoded_predictions = []
                for prediction in predictions:
                    if "one2many" in args.dataset:
                        decoded_prediction = qg_tokenizer.decode(prediction)
                        print("hello decoded prediction: ", decoded_prediction)
                        decoded_predictions_ = decoded_prediction.split("<sep>")
                        for decoded_prediction in decoded_predictions_:
                            decoded_predictions.append(decoded_prediction.strip())
                    else:
                        decoded_predictions.append(qg_tokenizer.decode(prediction))

                print("Document: ", document)
                print("Generated Questions: ", decoded_predictions)
                print("\n\n")

                predict_dict[document] = test_samples[id]
                predict_dict[document]["predictions"] = decoded_predictions
                predict_list.append(predict_dict[document])

    jsonlines_filename = Path("predictions/{}_{}_{}_{}.jsonl".format(args.dataset, args.model, args.model_type, time))
    Path("predictions/").mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fspath(jsonlines_filename), mode='w') as writer:
        writer.write_all(predict_list)




def run_and_collect_results(args, config):
    time = 0
    while time < args.times:
        run(args, config, time)
        time += 1


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    config = load_config(args)

    config["generate"] = True

    run_and_collect_results(args, config)
