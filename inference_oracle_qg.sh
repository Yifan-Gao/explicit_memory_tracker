#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

#PYT_QG=/research/king3/yfgao/miniconda3/envs/emtqg/bin/python
PYT_QG=/path/to/EMT_QG/bin/python

#DEV_PRED_PATH=pretrained_models #$2
#QG_MODEL_PATH=pretrained_models/qg.bin #$3

DEV_PRED_PATH=$2
QG_MODEL_PATH=$3

${PYT_QG} -u preprocess_qg.py \
--test \
--fpred=${DEV_PRED_PATH}

MODEL_DIR=pretrained_models
DATA_DIR=${DEV_PRED_PATH}
MODEL_RECOVER_PATH=${QG_MODEL_PATH}
export PYTORCH_PRETRAINED_BERT_CACHE=/tmp/"${MODEL_DIR}_cache/"

# run decoding
EVAL_SPLIT=oracleqg
OUTPUT_DIR=${DATA_DIR}/${EVAL_SPLIT}
FSRC='qg_src.txt'
${PYT_QG} qg/biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --cache_path=${MODEL_DIR} \
  --input_file ${DATA_DIR}/${FSRC} --split ${EVAL_SPLIT} \
  --output_file=${OUTPUT_DIR} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 256 --max_tgt_length 48 \
  --batch_size 4 --beam_size 10 --length_penalty 0

