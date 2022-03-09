from preprocess_tools.SQuADDuProcessor import process
from pathlib import Path
from os import fspath
import jsonlines
import pickle

for data_split in ["train", "dev", "test"]:
    para_filename = fspath(Path("../data/duetal/processed/para-{}.txt".format(data_split)))
    src_filename = fspath(Path("../data/duetal/processed/src-{}.txt".format(data_split)))
    tgt_filename = fspath(Path("../data/duetal/processed/tgt-{}.txt".format(data_split)))
    raw_filename = fspath(Path("../data/duetal/raw/{}.json".format(data_split)))

    sample_outputs, metadata_outputs = process(para_filename=para_filename,
                                               sentence_filename=src_filename,
                                               question_filename=tgt_filename,
                                               raw_filename=raw_filename)

    key = "_normal" if data_split in ["dev", "test"] else ""

    save_one2one_sentence_QG_path = fspath(
        Path("../processed_data/SQuADDu_one2one_sentence_QG_{}{}.jsonl".format(data_split, key)))
    save_one2many_sentence_QG_path = fspath(
        Path("../processed_data/SQuADDu_one2many_sentence_QG_{}{}.jsonl".format(data_split, key)))
    save_one2one_QG_path = fspath(
        Path("../processed_data/SQuADDu_one2one_QG_{}{}.jsonl".format(data_split, key)))
    save_one2many_QG_path = fspath(
        Path("../processed_data/SQuADDu_one2many_QG_{}{}.jsonl".format(data_split, key)))
    save_one2one_sentence_type_driven_QG_path = fspath(
        Path("../processed_data/SQuADDu_one2one_sentence_type_driven_QG_{}{}.jsonl".format(
            data_split, key)))
    save_one2many_sentence_type_driven_QG_path = fspath(
        Path("../processed_data/SQuADDu_one2many_sentence_type_driven_QG_{}{}.jsonl".format(
            data_split, key)))
    save_qw_classification_path = fspath(
        Path("../processed_data/SQuADDu_qw_{}{}.jsonl".format(data_split, key)))
    save_qw_hierarchical_labeling_path = fspath(
        Path("../processed_data/SQuADDu_qw_hierarchical_labeling_{}{}.jsonl".format(data_split, key)))
    save_qt_classification_path = fspath(
        Path("../processed_data/SQuADDu_qt_{}{}.jsonl".format(data_split, key)))

    save_one2one_sentence_QG_metadata_path = fspath(
        Path("../processed_data/SQuADDu_one2one_sentence_QG_metadata.pkl"))
    save_one2many_sentence_QG_metadata_path = fspath(
        Path("../processed_data/SQuADDu_one2many_sentence_QG_metadata.pkl"))
    save_one2one_QG_metadata_path = fspath(
        Path("../processed_data/SQuADDu_one2one_QG_metadata.pkl"))
    save_one2many_QG_metadata_path = fspath(
        Path("../processed_data/SQuADDu_one2many_QG_metadata.pkl"))
    save_one2one_sentence_type_driven_QG_metadata_path = fspath(
        Path("../processed_data/SQuADDu_one2one_sentence_type_driven_QG_metadata.pkl"))
    save_one2many_sentence_type_driven_QG_metadata_path = fspath(
        Path("../processed_data/SQuADDu_one2many_sentence_type_driven_QG_metadata.pkl"))
    save_qw_classification_metadata_path = fspath(
        Path("../processed_data/SQuADDu_qw_metadata.pkl"))
    save_qw_hierarchical_labeling_metadata_path = fspath(
        Path("../processed_data/SQuADDu_qw_hierarchical_labeling_metadata.pkl"))
    save_qt_classification_metadata_path = fspath(
        Path("../processed_data/SQuADDu_qt_metadata.pkl"))

    with jsonlines.open(save_qw_classification_path, mode='w') as writer:
        writer.write_all(sample_outputs["qw_classification"])

    with jsonlines.open(save_qw_hierarchical_labeling_path, mode='w') as writer:
        writer.write_all(sample_outputs["qw_hierarchical_labeling"])

    with jsonlines.open(save_qt_classification_path, mode='w') as writer:
        writer.write_all(sample_outputs["qt_classification"])

    with jsonlines.open(save_one2one_QG_path, mode='w') as writer:
        writer.write_all(sample_outputs["one2one_QG"])

    with jsonlines.open(save_one2many_QG_path, mode='w') as writer:
        writer.write_all(sample_outputs["one2many_QG"])

    with jsonlines.open(save_one2one_sentence_QG_path, mode='w') as writer:
        writer.write_all(sample_outputs["one2one_sentence_QG"])

    with jsonlines.open(save_one2many_sentence_QG_path, mode='w') as writer:
        writer.write_all(sample_outputs["one2many_sentence_QG"])

    with jsonlines.open(save_one2one_sentence_type_driven_QG_path, mode='w') as writer:
        writer.write_all(sample_outputs["one2one_sentence_type_driven_QG"])

    with jsonlines.open(save_one2many_sentence_type_driven_QG_path, mode='w') as writer:
        writer.write_all(sample_outputs["one2many_sentence_type_driven_QG"])

    if data_split == "train":

        for key in ["qw_classification", "qw_hierarchical_labeling", "qt_classification",
                    "one2one_QG", "one2many_QG",
                    "one2one_sentence_QG", "one2many_sentence_QG",
                    "one2one_sentence_type_driven_QG", "one2many_sentence_type_driven_QG"]:
            metadata = metadata_outputs[key]
            metadata["dev_keys"] = ["normal"]
            metadata["test_keys"] = ["normal"]
            metadata_save_path = eval("save_{}_metadata_path".format(key))

            with open(metadata_save_path, 'wb') as outfile:
                pickle.dump(metadata, outfile)
