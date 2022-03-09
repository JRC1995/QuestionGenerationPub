from os import fspath
from pathlib import Path
from nltk import sent_tokenize
from nltk import word_tokenize
import json
import string
import random
import numpy as np


def refine_tokenization(tokens):
    new_tokens = []
    for token in tokens:
        token = token.replace("`", "'")
        sub_tokens = [token]
        if token != "''":
            if token[0] in string.punctuation and len(token) > 1:
                if token[1] in string.punctuation:
                    if len(token) > 2:
                        sub_tokens = [token[0], token[1:-1], token[-1]]
                    else:
                        sub_tokens = [token[0], token[-1]]
                else:
                    sub_tokens = [token[0], token[1:]]
            elif len(token) > 1 and token[1] in string.punctuation:
                sub_tokens = [token[0:-1], token[-1]]
        new_tokens += sub_tokens

    return new_tokens


def question_type_mapper(filename):
    question2type = {}
    with open(filename) as fp:
        objs = json.load(fp)
        for obj in objs:
            paragraphs = obj["paragraphs"]
            for paragraph in paragraphs:
                context = paragraph["context"]
                qas = paragraph["qas"]
                for qa in qas:
                    answers = [obj["text"] for obj in qa["answers"]]
                    answer = answers[0].lower()
                    question = qa["question"].lower()
                    question = " ".join(refine_tokenization(word_tokenize(question)))

                    if answer == "yes" or answer == "no":
                        question2type[question] = "other"
                    else:
                        flag = 0
                        for type in ["who", "whom", "whose",
                                     "when", "where", "what",
                                     "how many", "how much",
                                     "how", "why", "which"]:
                            if (len(type.split(" ")) > 1 and type in question) \
                                    or (type in question.split(" ") and len(type.split(" ")) == 1):

                                if type == "how many" or type == "how much":
                                    type = "quantity"
                                if type == "whom" or type == "whose":
                                    type = "who"
                                question2type[question] = type
                                flag = 1
                                break
                        if flag == 0:
                            question2type[question] = "other"

    return question2type


def extract_lines(filename):
    lines_ = []
    with open(filename) as fp:
        lines = fp.readlines()
        for line in lines:
            lines_.append(line.strip().lower())
    return lines_


def question_type_map(question, question2type):
    question = " ".join(refine_tokenization(question.split(" ")))
    try:
        question_type = question2type[question]
        # print(question_type)
    except:
        flag = 0
        for type in ["who", "whom", "whose", "when", "where", "what", "how many", "how much", "how", "why",
                     "which"]:
            if (len(type.split(" ")) > 1 and type in question) \
                    or (type in question.split(" ") and len(type.split(" ")) == 1):
                if type == "how many" or type == "how much":
                    type = "quantity"
                if type == "whom" or type == "whose":
                    type = "who"
                question_type = type
                flag = 1
                break
        if flag == 0:
            question_type = "other"

    return question_type


def generate_question_worthiness_hierarchical_labeler_data(paragraphs, sentences):
    question_worthy_sentences = list(set(sentences))
    question_worthy_sentences = {sentence: 1 for sentence in question_worthy_sentences}
    all_paragraphs = []
    qw_labels = []
    for paragraph in list(set(paragraphs)):
        sentences = sent_tokenize(paragraph)
        all_paragraphs.append(sentences)
        qw_label = []
        for sentence in sentences:
            if sentence in question_worthy_sentences:
                qw_label.append(1)
            else:
                qw_label.append(0)
        qw_labels.append(qw_label)

    samples = []

    for paragraph, qw_label in zip(all_paragraphs, qw_labels):
        sample = {"paragraph": paragraph,
                  "label": qw_label}
        samples.append(sample)

    metadata = {"labels2idx": {"question_worthy": 0}}

    return samples, metadata


def generate_question_worthiness_classifier_data(paragraphs, sentences):
    question_worthy_sentences = list(set(sentences))
    question_worthy_sentences = {sentence: 1 for sentence in question_worthy_sentences}
    all_sentences = []
    qw_classes = []
    for paragraph in list(set(paragraphs)):
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            all_sentences.append(sentence)
            if sentence in question_worthy_sentences:
                qw_class = 1
            else:
                qw_class = 0
            qw_classes.append(qw_class)

    samples = []

    class_1_num = np.sum(qw_classes)
    class_0_num = len(qw_classes) - class_1_num
    max_num = max(class_1_num, class_0_num)

    label_weights = [class_0_num / class_1_num]

    for sentence, qw_class in zip(all_sentences, qw_classes):
        sample = {"sentence": sentence,
                  "class": qw_class}
        samples.append(sample)

    metadata = {"labels2idx": {"question_worthy": 0},
                "label_weights": label_weights}

    return samples, metadata


def generate_question_type_classifier_data(sentences, question_types, all_question_types):
    idx2labels = {id: qt for id, qt in enumerate(all_question_types)}
    sentences_dict = {}
    for sentence, question_type in zip(sentences, question_types):
        if sentence in sentences_dict:
            sentences_dict[sentence].append(question_type)
        else:
            sentences_dict[sentence] = [question_type]

    samples = []

    for sentence in sentences_dict:
        labels = []
        for id in range(len(idx2labels)):
            if idx2labels[id] in sentences_dict[sentence]:
                labels.append(1)
            else:
                labels.append(0)

        sample = {"sentence": sentence,
                  "class": labels,
                  "idx2labels": idx2labels}
        samples.append(sample)

    labels = [sample["class"] for sample in samples]
    pos_nums = np.sum(np.asarray(labels), axis=0)
    neg_nums = len(labels) - pos_nums
    max_nums = np.maximum(pos_nums, neg_nums)
    label_weights = neg_nums / pos_nums
    print(label_weights)

    metadata = {"label_weights": label_weights.tolist(),
                "labels2idx": {v: k for k, v in idx2labels.items()}}

    return samples, metadata


def generate_one2one_sentence_QG_data(sentences, questions, paragraphs):
    samples = []

    sentences_dict = {}
    for sentence, paragraph, question in zip(sentences, paragraphs, questions):
        if sentence in sentences_dict:
            sentences_dict[sentence].append(question)
        else:
            sentences_dict[sentence] = [question]

    for sentence, paragraph, question in zip(sentences, paragraphs, questions):
        sample = {"question": question,
                  "questions": sentences_dict[sentence],
                  "document": paragraph.replace(sentence, " <hl> " + sentence + " </hl> ")}
        samples.append(sample)

    return samples, {}


def generate_one2many_sentence_QG_data(sentences, questions, paragraphs):
    samples = []
    sentences_dict = {}

    for sentence, paragraph, question in zip(sentences, paragraphs, questions):
        if sentence in sentences_dict:
            sentences_dict[sentence]["question"].append(question)
        else:
            sentences_dict[sentence] = {"paragraph": paragraph, "question": [question]}

    for sentence in sentences_dict:
        sample = {"question": " <sep> ".join(sentences_dict[sentence]["question"]),
                  "questions": sentences_dict[sentence]["question"],
                  "document": sentences_dict[sentence]["paragraph"].replace(sentence,
                                                                            " <hl> " + sentence + " </hl> ")}
        samples.append(sample)

    return samples, {}


def generate_one2many_QG_data(paragraphs, questions):
    samples = []
    paragraphs_dict = {}
    for paragraph, question in zip(paragraphs, questions):
        if paragraph in paragraphs_dict:
            paragraphs_dict[paragraph].append(question)
        else:
            paragraphs_dict[paragraph] = [question]

    question_nums = []

    for paragraph in paragraphs_dict:
        question_nums.append(len(paragraphs_dict[paragraph]))
        sample = {"question": " <sep> ".join(paragraphs_dict[paragraph]),
                  "questions": paragraphs_dict[paragraph],
                  "document": paragraph}
        samples.append(sample)

    print("average_number of questions: ", np.mean(question_nums))

    return samples, {}


def generate_one2one_QG_data(paragraphs, questions):
    samples = []
    paragraph2questions = {}
    for paragraph, question in zip(paragraphs, questions):
        if paragraph in paragraph2questions:
            paragraph2questions[paragraph].append(question)
        else:
            paragraph2questions[paragraph] = [question]

    for paragraph, question in zip(paragraphs, questions):
        sample = {"question": question,
                  "questions": paragraph2questions[paragraph],
                  "document": paragraph}
        samples.append(sample)

    return samples, {}


def generate_one2many_sentence_type_driven_QG_data(sentences, questions, question_types, paragraphs):
    samples = []
    sentences_dict = {}
    for sentence, paragraph, question, question_type in zip(sentences, paragraphs, questions, question_types):
        if sentence in sentences_dict:
            if question_type in sentences_dict[sentence]:
                sentences_dict[sentence]["question"][question_type].append(question)
            else:
                sentences_dict[sentence]["question"][question_type] = [question]
        else:
            sentences_dict[sentence] = {"paragraph": paragraph, "question": {question_type: [question]}}

    for sentence in sentences_dict:
        for question_type in sentences_dict[sentence]["question"]:
            sample = {"question": " <sep> ".join(sentences_dict[sentence]["question"][question_type]),
                      "questions": sentences_dict[sentence]["question"][question_type],
                      "document": "<" + question_type + "> " + sentences_dict[sentence]["paragraph"].replace(sentence,
                                                                                                             " <hl> " + sentence + " </hl> ")}
            samples.append(sample)

    return samples, {}


def generate_one2one_sentence_type_driven_QG_data(sentences, questions, question_types, paragraphs):
    samples = []
    sentences_dict = {}
    for sentence, paragraph, question, question_type in zip(sentences, paragraphs, questions, question_types):
        if sentence in sentences_dict:
            if question_type in sentences_dict[sentence]:
                sentences_dict[sentence]["question"][question_type].append(question)
            else:
                sentences_dict[sentence]["question"][question_type] = [question]
        else:
            sentences_dict[sentence] = {"paragraph": paragraph, "question": {question_type: [question]}}

    for sentence, paragraph, question, question_type in zip(sentences, paragraphs, questions, question_types):
        for question_type in sentences_dict[sentence]["question"]:
            sample = {"question": question,
                      "questions": sentences_dict[sentence]["question"][question_type],
                      "document": "<" + question_type + "> " + sentences_dict[sentence]["paragraph"].replace(sentence,
                                                                                                             " <hl> " + sentence + " </hl> ")}
            samples.append(sample)

    return samples, {}


def process(para_filename, sentence_filename, question_filename, raw_filename):
    all_question_types = ["who",
                          "when", "where", "what",
                          "quantity",
                          "how", "why", "which", "other"]

    question2type = question_type_mapper(raw_filename)
    paragraphs = extract_lines(para_filename)
    questions = extract_lines(question_filename)
    sentences = extract_lines(sentence_filename)
    question_types = []
    for question in questions:
        question_type = question_type_map(question, question2type)
        question_types.append(question_type)

    qw_hierarchical_labeling_samples, qw_hierarchical_labeling_metadata \
        = generate_question_worthiness_hierarchical_labeler_data(paragraphs=paragraphs,
                                                                 sentences=sentences)

    qw_classification_samples, qw_classification_metadata \
        = generate_question_worthiness_classifier_data(paragraphs=paragraphs,
                                                       sentences=sentences)
    qt_classification_samples, qt_classification_metadata \
        = generate_question_type_classifier_data(sentences=sentences,
                                                 question_types=question_types,
                                                 all_question_types=all_question_types)
    one2one_QG_samples, one2one_QG_metadata \
        = generate_one2one_QG_data(paragraphs=paragraphs,
                                   questions=questions)

    one2many_QG_samples, one2many_QG_metadata \
        = generate_one2many_QG_data(paragraphs=paragraphs,
                                    questions=questions)

    one2one_sentence_QG_samples, one2one_sentence_QG_metadata \
        = generate_one2one_sentence_QG_data(sentences=sentences,
                                            questions=questions,
                                            paragraphs=paragraphs)
    one2many_sentence_QG_samples, one2many_sentence_QG_metadata \
        = generate_one2many_sentence_QG_data(sentences=sentences,
                                             questions=questions,
                                             paragraphs=paragraphs)

    one2one_sentence_type_driven_QG_samples, one2one_sentence_type_driven_QG_metadata \
        = generate_one2one_sentence_type_driven_QG_data(sentences=sentences,
                                                        questions=questions,
                                                        question_types=question_types,
                                                        paragraphs=paragraphs)

    one2many_sentence_type_driven_QG_samples, one2many_sentence_type_driven_QG_metadata \
        = generate_one2many_sentence_type_driven_QG_data(sentences=sentences,
                                                         questions=questions,
                                                         question_types=question_types,
                                                         paragraphs=paragraphs)

    for sample in qw_hierarchical_labeling_samples[0:100]:
        print(sample)


    samples_dict = {"qw_classification": qw_classification_samples,
                    "qw_hierarchical_labeling": qw_hierarchical_labeling_samples,
                    "qt_classification": qt_classification_samples,
                    "one2one_QG": one2one_QG_samples,
                    "one2many_QG": one2many_QG_samples,
                    "one2one_sentence_QG": one2one_sentence_QG_samples,
                    "one2many_sentence_QG": one2many_sentence_QG_samples,
                    "one2one_sentence_type_driven_QG": one2one_sentence_type_driven_QG_samples,
                    "one2many_sentence_type_driven_QG": one2many_sentence_type_driven_QG_samples}

    metadata_dict = {"qw_classification": qw_classification_metadata,
                     "qw_hierarchical_labeling": qw_hierarchical_labeling_metadata,
                     "qt_classification": qt_classification_metadata,
                     "one2one_QG": one2one_QG_metadata,
                     "one2many_QG": one2many_QG_metadata,
                     "one2one_sentence_QG": one2one_sentence_QG_metadata,
                     "one2many_sentence_QG": one2many_sentence_QG_metadata,
                     "one2one_sentence_type_driven_QG": one2one_sentence_type_driven_QG_metadata,
                     "one2many_sentence_type_driven_QG": one2many_sentence_type_driven_QG_metadata}

    return samples_dict, metadata_dict
