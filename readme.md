## Requirements

Install ``https://github.com/Maluuba/nlg-eval``

For the rest, Install from requirements.txt

## Credits

RAdam code is adapted from: https://github.com/LiyuanLucasLiu/RAdam
Ranger code is adapted from: https://github.com/anoidgit/transformer/blob/master/optm/ranger.py
SM3 code is adapted from: https://github.com/Enealor/PyTorch-SM3
transformers_custom is adapted from Huggingface. 

## Download embeddings and process_data 

Download embeddings through preprocess/t5_download_qg.py and preprocess/electra_download.py


Keep [SQuAD Du et al. data split](https://github.com/xinyadu/nqg/tree/master/data) in data/duetal/


Preprocess it through preprocess/process_du.py


## Train Question Worthiness Classifier on SQuAD

``python train.py --model=ELECTRAHierarchicalLabeler --dataset=SQuADDu_qw_hierarchical_labeling --model_type=HierarchicalLabeler``

Copy paste the final accuracy and everything from the terminal. 

## Train Question Type Classifier on SQuAD

``python train.py --model=ELECTRAMultiLabelClassifier --dataset=SQuADDu_qt --model_type=BinaryClassifier``

Copy paste the final accuracy and everything from the terminal. 


## train one2one 

``python train.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_QG  --model_type=Seq2Seq --lr=X``

(X = hypertuned lr for the model)

## predict one2one (after completing training)

``python predict.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_QG --model_type=Seq2Seq``

## train one2many

``python train.py --model=T5Seq2Seq --dataset=SQuADDu_one2many_QG  --model_type=Seq2Seq``

## predict one2many (after completing training)

``python predict.py --model=T5Seq2Seq --dataset=SQuADDu_one2many_QG --model_type=Seq2Seq``

(X = hypertuned lr for the model)

## train one2one sentence level 

``python train.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_sentence_QG --model_type=Seq2Seq``

## predict one2one sentence level (after completing training)

``python predict.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_sentence_QG --model_type=Seq2Seq``

## train one2many sentence level

``python train.py --model=T5Seq2Seq --dataset=SQuADDu_one2many_sentence_QG --model_type=Seq2Seq``

## predict one2many sentence level (after completing training)

``python predict.py --model=T5Seq2Seq --dataset=SQuADDu_one2many_sentence_QG --model_type=Seq2Seq``

## train one2one type level 

``python train.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_sentence_type_driven_QG --model_type=Seq2Seq``

## predict one2one type level (after completing training)

``python predict.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_sentence_type_driven_QG --model_type=Seq2Seq``

## train one2many type level

``python train.py --model=T5Seq2Seq --dataset=SQuADDu_one2many_sentence_type_driven_QG --model_type=Seq2Seq``

## predict one2many type level (after completing training)

``python predict.py --model=T5Seq2Seq --dataset=SQuADDu_one2many_sentence_type_driven_QG --model_type=Seq2Seq``

## Use advanced rankers on one2one predictions

``python rank_predictions.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_QG --model_type=Seq2Seq``
``python rank_predictions.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_sentence_QG --model_type=Seq2Seq``
``python rank_predictions.py --model=T5Seq2Seq --dataset=SQuADDu_one2one_sentence_type_driven_QG --model_type=Seq2Seq``

## Run Evaluation

``bash SQuADDu_qg_eval.sh``
