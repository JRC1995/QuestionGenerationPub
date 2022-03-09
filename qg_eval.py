from tokenizer import PTBTokenizer
from answerability_score import get_answerability_score
from argparse import ArgumentParser
import argparse
from munkres import Munkres
from nlgeval import NLGEval
from pathlib import Path
import jsonlines
import copy
import numpy as np
from tqdm import tqdm
import yaml
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="QG EVALUATION")

    parser.add_argument('--model', type=str, default="T5Seq2Seq",
                        choices=["T5Seq2Seq",
                                 "T5Seq2SeqVAE"])
    parser.add_argument('--model_type', type=str, default="Seq2Seq",
                        choices=["Seq2Seq"])
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--top1', type=str2bool, default=False)
    parser.add_argument('--reference_self_bleu', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default="SQuADDu_one2one_QG")
    return parser


parser = get_args()
args = parser.parse_args()
metrics_to_omit = ['CIDEr', 'EmbeddingAverageCosineSimilarity', 'SkipThoughtCS', 'VectorExtremaCosineSimilarity',
                   'GreedyMatchingScore']
Path('evaluations/').mkdir(parents=True, exist_ok=True)

tokenizer = PTBTokenizer()
m = Munkres()
delta = 0.66
nlgeval = NLGEval(metrics_to_omit=metrics_to_omit)

open_filename = Path("predictions/{}_{}_{}_{}.jsonl".format(args.dataset, args.model, args.model_type, args.time))

sets_of_questions = []
sets_of_predictions = []
documents = []
with jsonlines.open(open_filename, "r") as Reader:
    for id, obj in enumerate(Reader):
        sets_of_predictions.append(obj["predictions"])
        sets_of_questions.append(obj["questions"])
        documents.append(obj["document"])

if "one2one" in args.dataset and args.top1:
    new_sets_of_predictions = []
    for predictions in sets_of_predictions:
        new_predictions = []
        if "sentence" in args.dataset:
            k = 10
        elif "type" in args.dataset:
            k = 5
        else:
            k = 20
        for i, prediction in enumerate(predictions):
            if i % k == 0:
                new_predictions.append(prediction)
        new_sets_of_predictions.append(new_predictions)
    sets_of_predictions = new_sets_of_predictions

sets_of_predictions = [list(set(predictions)) for predictions in sets_of_predictions]


def nested_tokenize(nested_list):
    list_lens = [len(items) for items in nested_list]
    flat_list = []
    for items in nested_list:
        flat_list += items
    flat_list = tokenizer.tokenize(flat_list)
    nested_list = []
    i = 0
    for length in list_lens:
        nested_list.append(flat_list[i:i + length])
        i += length

    return nested_list


documents = tokenizer.tokenize(documents)
sets_of_predictions = nested_tokenize(sets_of_predictions)
sets_of_questions = nested_tokenize(sets_of_questions)

if "one2one" in args.dataset and "ranked" not in args.dataset and not args.top1:
    ref_open_filename = Path("predictions/{}_{}_{}_{}.jsonl".format(args.dataset.replace("one2one", "one2many"),
                                                                    "T5Seq2Seq", args.model_type, args.time))
    doc2pred_num = {}
    new_sets_of_predictions = []
    with jsonlines.open(ref_open_filename, "r") as Reader:
        for id, obj in enumerate(Reader):
            new_sets_of_predictions.append(obj["predictions"])

    for document, predictions in zip(documents, new_sets_of_predictions):
        doc2pred_num[document] = max(1, len(set(predictions)))
else:
    doc2pred_num = {}
    for document, predictions in zip(documents, sets_of_predictions):
        doc2pred_num[document] = max(1, len(set(predictions)))

select_types = ["@M"]
if "one2one" in args.dataset and not args.top1 and "ranked" not in args.dataset:
    select_types = ["best@G", "best@M", "best@5", "rand@G", "rand@M", "rand@5"]


def compute_optimal_assignment(profit_matrix):
    global m
    profit_matrix_ = copy.deepcopy(profit_matrix)
    np_profit_matrix = np.asarray(profit_matrix_)
    # converting profit matrix to meet the pre-conditions of the hungarian algorithm code
    np_positive_profit_matrix = np_profit_matrix - max(0, np.min(np_profit_matrix))
    np_rounded_profit_matrix = np.around(np_positive_profit_matrix * 100000).astype(int)
    np_cost_matrix = (np.max(np_rounded_profit_matrix) + 1) - np_rounded_profit_matrix
    input_cost_matrix = np_cost_matrix.tolist()
    # print("modified cost matrix: ", input_cost_matrix)
    optimal_indexes = m.compute(input_cost_matrix)
    return optimal_indexes


def file_write(text, file):
    file.write(str(text.encode('utf-8').decode('utf-8')))


for select_type in select_types:
    save_global_filename = Path("evaluations/{}_{}_{}_{}_{}_top1{}.yaml".format(args.dataset, select_type, args.model,
                                                                                args.model_type, args.time, args.top1))
    save_individual_filename = Path("evaluations/{}_{}_{}_{}_{}_top1{}.txt".format(args.dataset, select_type,
                                                                                   args.model, args.model_type,
                                                                                   args.time, args.top1))

    global_metrics = {}
    j = 0
    with open(save_individual_filename, 'w', encoding='utf-8') as file:
        with tqdm(total=len(documents),
                  desc="Evaluating {} {} {}".format(args.dataset, args.model, select_type), position=0) as pbar:
            for predictions, questions, document in zip(sets_of_predictions, sets_of_questions, documents):
                file_write("Document #{}: {}".format(j, document) + "\n" + "---------" + "\n\n", file)
                j += 1
                file_write("Reference Questions:\n", file)
                for question in questions:
                    file_write(question + "\n", file)
                file_write("---------" + "\n\n", file)

                file_write("All Predictions:\n", file)
                for prediction in predictions:
                    file_write(prediction + "\n", file)
                file_write("---------" + "\n\n", file)

                individual_metrics = {}
                profit_matrix = {}

                if not predictions:
                    predictions = ["<DUMMYSTUFFSURETOACHIEVE0>"]
                if not questions:
                    questions = ["<DUMMYSTUFFSURETOACHIEVE0>"]

                if "@5" in select_type:
                    pred_num = min(len(predictions), 5)
                elif "@G" in select_type:
                    pred_num = min(len(predictions), len(questions))
                else:
                    pred_num = min(len(predictions), doc2pred_num[document])

                for prediction in predictions:
                    row = {}
                    for question in questions:
                        metrics = nlgeval.compute_individual_metrics([question], prediction)
                        ans_score = get_answerability_score(prediction, question)
                        metrics["q-BLEU1"] = delta * ans_score + (1 - delta) * copy.deepcopy(metrics["Bleu_1"])
                        metrics.pop("Bleu_1")
                        metrics.pop("Bleu_2")
                        metrics.pop("Bleu_3")
                        for key in metrics:
                            if key in row:
                                row[key].append(metrics[key])
                            else:
                                row[key] = [metrics[key]]
                    for key in metrics:
                        if key in profit_matrix:
                            profit_matrix[key].append(row[key])
                        else:
                            profit_matrix[key] = [row[key]]

                if "best" in select_type:
                    optimal_indexes = {}
                    optimal_indexes["METEOR"] = compute_optimal_assignment(profit_matrix["METEOR"])

                    # select based on METEOR
                    vals = []
                    assigned_pred_idx = []
                    for row, column in optimal_indexes["METEOR"]:
                        vals.append(profit_matrix[key][row][column])
                        assigned_pred_idx.append(row)
                    sorted_idx = np.flip(np.argsort(vals), axis=-1).tolist()
                    best_pred_idx = [assigned_pred_idx[i] for i in sorted_idx]

                    if len(best_pred_idx) < len(predictions):
                        other_pred_idx = [id for id in range(len(predictions)) if id not in best_pred_idx]
                        other_vals = []
                        for id in other_pred_idx:
                            val = max(profit_matrix["METEOR"][id])
                            other_vals.append(val)
                        sorted_idx = np.flip(np.argsort(other_vals), axis=-1)
                        other_pred_idx = [other_pred_idx[id] for id in sorted_idx]
                        best_pred_idx = best_pred_idx + other_pred_idx

                    ranked_idx = best_pred_idx[0:pred_num]
                elif "rand" in select_type:
                    ranked_idx = [id for id in range(len(predictions))]
                    ranked_idx = random.sample(ranked_idx, k=pred_num)
                else:
                    ranked_idx = [id for id in range(len(predictions))]
                    ranked_idx = ranked_idx[0:pred_num]

                file.write(
                    "Predictions Selected According to Best METEOR Score after optimal assignment to Reference Questions " \
                    + "(based on selection method {}):\n".format(select_type))
                for i in ranked_idx:
                    file_write(predictions[i] + "\n", file)
                file.write("---------" + "\n\n")

                profit_matrix_ = {}
                optimal_indexes = {}
                for key in profit_matrix:
                    profit_matrix[key] = [profit_matrix[key][row] for row in ranked_idx]
                    optimal_indexes[key] = compute_optimal_assignment(profit_matrix[key])

                file.write("Assignments to {} (based on METEOR) predictions:\n".format(select_type))
                for key in optimal_indexes:
                    file.write("\n\n({})\n\n".format(key))
                    vals = []
                    for row, column in optimal_indexes[key]:
                        vals.append(profit_matrix[key][row][column])
                        file_write("prediction: {}\n".format(predictions[ranked_idx[row]]), file)
                        file_write("Assigned Reference: {}\n".format(questions[column]), file)
                        file_write("{}: {}\n\n".format(key, profit_matrix[key][row][column]), file)
                    file.write("---------" + "\n\n")

                    individual_metrics["multi_" + key + "_prec"] = sum(vals) / pred_num if pred_num != 0 else 0
                    individual_metrics["multi_" + key + "_rec"] = sum(vals) / len(questions) \
                        if len(questions) != 0 else 0
                    p = individual_metrics["multi_" + key + "_prec"]
                    r = individual_metrics["multi_" + key + "_rec"]
                    individual_metrics["multi_" + key + "_F1"] = (2 * p * r) / (p + r) if (p + r) != 0 else 0

                sub_metrics = {}
                for prediction in predictions:
                    metrics = nlgeval.compute_individual_metrics(questions, prediction)
                    ans_score = max(get_answerability_score(prediction, question) for question in questions)
                    metrics["q-BlEU1"] = delta * ans_score + (1 - delta) * copy.deepcopy(metrics["Bleu_1"])
                    metrics.pop("Bleu_1")
                    metrics.pop("Bleu_2")
                    metrics.pop("Bleu_3")
                    rest_predictions = [prediction_ for prediction_ in predictions if prediction_ != prediction]
                    if rest_predictions:
                        metrics["self-BLEU2"] = nlgeval.compute_individual_metrics(rest_predictions,
                                                                                   prediction)["Bleu_2"]
                    else:
                        metrics["self-BLEU2"] = 0
                    for key in metrics:
                        if key not in sub_metrics:
                            sub_metrics[key] = [metrics[key]]
                        else:
                            sub_metrics[key].append(metrics[key])

                for key in sub_metrics:
                    vals = sub_metrics[key]
                    select_vals = [vals[i] for i in ranked_idx]
                    individual_metrics["avg_" + key] \
                        = sum(select_vals) / pred_num if pred_num != 0 else 0

                if args.reference_self_bleu:
                    vals = 0
                    for question in questions:
                        rest_questions = [question_ for question_ in questions if question_ != question]
                        if rest_questions:
                            vals += nlgeval.compute_individual_metrics(rest_questions,
                                                                       question)["Bleu_2"]
                    individual_metrics["avg_ref_self-BLEU2"] = vals / len(questions) if len(questions) != 0 else 0

                individual_metrics["len_diff"] = len(questions) - pred_num

                file.write("Metrics:\n")
                for key in individual_metrics:
                    file.write("{}: {}\n".format(key, individual_metrics[key]))

                file.write("---------" + "\n\n")

                for key in individual_metrics:
                    if key not in global_metrics:
                        global_metrics[key] = individual_metrics[key]
                    else:
                        global_metrics[key] += individual_metrics[key]

                pbar.update(1)

    for key in global_metrics:
        global_metrics[key] /= len(sets_of_predictions)

    for key in global_metrics:
        global_metrics[key] = float(global_metrics[key])

    with open(save_global_filename, "w") as fp:
        yaml.dump(global_metrics, fp)
