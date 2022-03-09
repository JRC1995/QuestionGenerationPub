import random
import zlib
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch.nn as nn
from models.transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch as T
import torch.nn.functional as F
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
from models.transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import copy
from tqdm import tqdm

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

model_name = "ahotrod/electra_large_discriminator_squad2_512"
time = 0
parser = get_args()
args = parser.parse_args()
config = load_config(args)

open_filename = Path("predictions/{}_{}_{}_{}.jsonl".format(args.dataset, args.model, args.model_type, time))

sets_of_questions = []
sets_of_predictions = []
documents = []
with jsonlines.open(open_filename, "r") as Reader:
    for id, obj in enumerate(Reader):
        sets_of_predictions.append(list(set(obj["predictions"])))
        sets_of_questions.append(obj["questions"])
        documents.append(obj["document"])

paramodel = SentenceTransformer('paraphrase-mpnet-base-v2').cuda()
all_embeddings = [paramodel.encode(predictions, convert_to_tensor=True) if predictions else None \
                  for predictions in sets_of_predictions]


def hm(f1, f2):
    return ((2 * f1 * f2) / (f1 + f2)) if (f1 + f2) != 0 else 0


# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_impossibility_scores(questions, contexts):
    batch_input_ids = []
    batch_token_type = []
    batch_attn_mask = []
    for question, context in zip(questions, contexts):
        input_dict = tokenizer.encode_plus(text=question, text_pair=context, add_special_tokens=True,
                                           max_length=512, truncation=True)
        batch_input_ids.append(input_dict["input_ids"])
        batch_attn_mask.append(input_dict["attention_mask"])
        if "token_type_ids" in input_dict:
            batch_token_type.append((input_dict["token_type_ids"]))
        else:
            batch_token_type = None

    max_len = max([len(sample) for sample in batch_input_ids])

    for i in range(len(batch_input_ids)):
        sample = batch_input_ids[i]
        if len(sample) < max_len:
            sample = sample + [tokenizer.pad_token_id] * (max_len - len(sample))
            batch_input_ids[i] = sample
            sample = batch_attn_mask[i]
            sample = sample + [0] * (max_len - len(sample))
            batch_attn_mask[i] = sample
            if batch_token_type is not None:
                sample = batch_token_type[i]
                sample = sample + [0] * (max_len - len(sample))
                batch_token_type[i] = sample

    if batch_token_type is not None:
        batch_token_type = T.tensor(batch_token_type).cuda()
    batch_input_ids = T.tensor(batch_input_ids).cuda()
    batch_attn_mask = T.tensor(batch_attn_mask).cuda()

    with T.no_grad():
        out_dict = model(input_ids=batch_input_ids,
                         attention_mask=batch_attn_mask,
                         token_type_ids=batch_token_type,
                         return_dict=False)
        start_logits = out_dict[0]
        end_logits = out_dict[1]

        start_logits = start_logits * batch_attn_mask + (1 - batch_attn_mask) * -100000
        end_logits = end_logits * batch_attn_mask + (1 - batch_attn_mask) * -100000

        N, S = start_logits.size()

        question_mask = T.cat([T.ones(N, 1).float().to(batch_token_type.device),
                               batch_token_type[:, 1:].float()], dim=1)

        start_logits = start_logits * question_mask + (1 - question_mask) * -100000
        end_logits = end_logits * question_mask + (1 - question_mask) * -100000

        start_logits = F.softmax(start_logits, dim=-1)
        end_logits = F.softmax(end_logits, dim=-1)

        impossibility_scores = start_logits[:, 0] * end_logits[:, 0]
        impossibility_scores = impossibility_scores.detach().cpu().numpy().tolist()

        return impossibility_scores


sets_of_simple_ranked_predictions = []
sets_of_complicated_ranked_predictions5 = []
sets_of_complicated_ranked_predictionsM = []

with tqdm(total=len(documents), desc="ranking", position=0) as pbar:
    for predictions, questions, document, embeddings in zip(sets_of_predictions, sets_of_questions, documents,
                                                            all_embeddings):

        """                                                   
        QA_inputs = []
        for prediction in predictions:
            QA_inputs.append({"question": prediction, "context": document})
    
        print("document: ", document)
        print("predictions: ", predictions)
        _, impossibility_scores = nlp(QA_inputs, handle_impossible_answer=True, device='cuda')
        """
        # print("document: ", document)
        # print("predictions: ", predictions)
        if not predictions:
            sets_of_simple_ranked_predictions.append([])
            sets_of_complicated_ranked_predictions5.append([])
            sets_of_complicated_ranked_predictionsM.append([])
        else:
            contexts = [document] * len(predictions)
            impossibility_scores = get_impossibility_scores(questions=predictions, contexts=contexts)
            answerability_scores = [1 - score for score in impossibility_scores]

            rank_ids = np.flip(np.argsort(answerability_scores), axis=-1)
            # print("sorted_scores: ", [answerability_scores[i] for i in rank_ids])
            rank_ids = [id for id in rank_ids if impossibility_scores[id] < 0.5]  # filtering impossible questionsW
            ranked_predictions = [predictions[id] for id in rank_ids]
            ranked_scores = [answerability_scores[id] for id in rank_ids]

            # print("initial ranked predictions: ", ranked_predictions)

            sets_of_simple_ranked_predictions.append(ranked_predictions[0:5])

            complicated_ranked_predictions = []
            query_embeddings = []
            M = 0
            if ranked_predictions:
                ranked_embeddings = [embeddings[i] for i in rank_ids]
                complicated_ranked_predictions.append(copy.deepcopy(ranked_predictions[0]))
                query_embeddings.append(copy.deepcopy(ranked_embeddings[0]))
                del ranked_predictions[0]
                del ranked_embeddings[0]
                del ranked_scores[0]
                M += 1

            while ranked_predictions:
                cosine_scores = util.pytorch_cos_sim(T.stack(query_embeddings, dim=0),
                                                     T.stack(ranked_embeddings, dim=0))
                assert cosine_scores.size() == (len(query_embeddings), len(ranked_embeddings))
                cosine_scores = T.max(cosine_scores, dim=0)[0].detach().cpu().tolist()
                assert len(cosine_scores) == len(ranked_predictions)
                assert len(ranked_scores) == len(cosine_scores)
                inverse_cosine_scores = [1 - max(0, f) for f in cosine_scores]
                # print("dissimilarity scores: ", inverse_cosine_scores)
                # print("answerability scores: ", ranked_scores)
                # print("selected predictions: ", complicated_ranked_predictions)
                # print("unselected predictions: ", ranked_predictions)
                hm_scores = [hm(f1, f2) for f1, f2 in zip(inverse_cosine_scores, ranked_scores)]
                new_rank_ids = np.flip(np.argsort(hm_scores), axis=-1)
                chosen_id = new_rank_ids[0]
                complicated_ranked_predictions.append(copy.deepcopy(ranked_predictions[chosen_id]))
                query_embeddings.append(copy.deepcopy(ranked_embeddings[chosen_id]))
                del ranked_embeddings[chosen_id]
                del ranked_predictions[chosen_id]
                del ranked_scores[chosen_id]
                if max(inverse_cosine_scores) > 0.2:
                    M += 1

            complicated_ranked_predictions5 = complicated_ranked_predictions[0:5]
            complicated_ranked_predictionsM = complicated_ranked_predictions[0:M]

            # print("complicated ranked predictions @5: ", complicated_ranked_predictions5)
            # print("complicated_ranked_predictions @M: ", complicated_ranked_predictionsM)

            sets_of_complicated_ranked_predictions5.append(complicated_ranked_predictions5)
            sets_of_complicated_ranked_predictionsM.append(complicated_ranked_predictionsM)
        pbar.update(1)

print(documents)

predict_list_simple = []
predict_list_complicated5 = []
predict_list_complicatedM = []
for predictions, questions, document, simple_ranked_predictions, \
    complicated_ranked_predictions5, complicated_ranked_predictionsM in zip(sets_of_predictions, sets_of_questions,
                                                                            documents,
                                                                            sets_of_simple_ranked_predictions,
                                                                            sets_of_complicated_ranked_predictions5,
                                                                            sets_of_complicated_ranked_predictionsM):
    simple_obj = {"document": document,
                  "questions": questions,
                  "predictions": simple_ranked_predictions}

    complicated5_obj = {"document": document,
                        "questions": questions,
                        "predictions": complicated_ranked_predictions5}

    complicatedM_obj = {"document": document,
                        "questions": questions,
                        "predictions": complicated_ranked_predictionsM}

    predict_list_simple.append(simple_obj)
    predict_list_complicated5.append(complicated5_obj)
    predict_list_complicatedM.append(complicatedM_obj)

jsonlines_filename = Path(
    "predictions/{}_{}_{}_{}.jsonl".format(args.dataset + "_simple_ranked", args.model, args.model_type, time))
Path("predictions/").mkdir(parents=True, exist_ok=True)
with jsonlines.open(fspath(jsonlines_filename), mode='w') as writer:
    writer.write_all(predict_list_simple)

jsonlines_filename = Path(
    "predictions/{}_{}_{}_{}.jsonl".format(args.dataset + "_advanced_ranked5", args.model, args.model_type, time))
Path("predictions/").mkdir(parents=True, exist_ok=True)
with jsonlines.open(fspath(jsonlines_filename), mode='w') as writer:
    writer.write_all(predict_list_complicated5)

jsonlines_filename = Path(
    "predictions/{}_{}_{}_{}.jsonl".format(args.dataset + "_advanced_rankedM", args.model, args.model_type, time))
Path("predictions/").mkdir(parents=True, exist_ok=True)
with jsonlines.open(fspath(jsonlines_filename), mode='w') as writer:
    writer.write_all(predict_list_complicatedM)
