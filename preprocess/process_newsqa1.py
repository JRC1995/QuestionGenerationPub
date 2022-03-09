import json
from nltk.tokenize import sent_tokenize
from pathlib import Path


def check_if_impossible(qa_obj):
    consensus = qa_obj["consensus"]
    if 'noAnswer' in consensus or 'badQuestion' in consensus:
        return False
    else:
        return True


def process(filename):
    train_srcs = []
    train_paras = []
    train_trgs = []

    dev_srcs = []
    dev_paras = []
    dev_trgs = []

    test_srcs = []
    test_paras = []
    test_trgs = []

    with open(filename, "r", encoding='utf-8') as fp:

        all_stuffs = json.load(fp)
        data = all_stuffs["data"]
        questions = {}

        for obj in data:

            document = obj["text"].lower()

            if "--" in document:
                document_ = "--".join(document.split("--")[1:])
            else:
                document_ = document

            document_ = document_.replace("\n", " ")
            sentences = sent_tokenize(document_)
            paragraph = "--|--".join(sentences)

            type = obj["type"]
            qas = obj["questions"]

            srcs = []
            trgs = []
            paras = []

            for qa_obj in qas:
                question = qa_obj["q"].strip().lower()
                if check_if_impossible(qa_obj) and question not in questions:
                    questions[question] = 1
                    consensus = qa_obj["consensus"]
                    start_id = consensus["s"]
                    end_id = consensus["e"]
                    answer = document[start_id:end_id].strip().lower()

                    answer_sentences = [sentence for sentence in sentences if answer in sentence]
                    if answer_sentences:
                        """
                        as_id = sentences.index(answer_sentence)
                        if as_id > 2:
                            selected_sentences = [sentences[as_id-2], sentences[as_id-1]] + sentences[as_id:]
                        else:
                            selected_sentences = sentences
                        """
                        for answer_sentence in answer_sentences:
                            if answer_sentence.strip(" ") != "" and question.strip(" ") != ""\
                                and answer_sentence in paragraph:
                                paras.append(paragraph + "\n")
                                srcs.append(answer_sentence + "\n")
                                trgs.append(question + "\n")

            if type == "test":
                test_paras += paras
                test_srcs += srcs
                test_trgs += trgs
            elif type == "dev":
                dev_paras += paras
                dev_srcs += srcs
                dev_trgs += trgs
            else:
                train_paras += paras
                train_srcs += srcs
                train_trgs += trgs

    with open(Path("../data/newsqa/para-train.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(train_paras)

    with open(Path("../data/newsqa/src-train.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(train_srcs)

    with open(Path("../data/newsqa/tgt-train.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(train_trgs)

    with open(Path("../data/newsqa/para-dev.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(dev_paras)

    with open(Path("../data/newsqa/src-dev.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(dev_srcs)

    with open(Path("../data/newsqa/tgt-dev.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(dev_trgs)

    with open(Path("../data/newsqa/para-test.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(test_paras)

    with open(Path("../data/newsqa/src-test.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(test_srcs)

    with open(Path("../data/newsqa/tgt-test.txt"), "w", encoding='utf-8') as fp:
        fp.writelines(test_trgs)


process(Path("../data/newsqa/combined-newsqa-data-v1.json"))
