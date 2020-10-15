#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1

#PYT_QG=/research/king3/yfgao/miniconda3/envs/emtqg/bin/python
PYT_QG=/path/to/EMT_QG/bin/python
DM_MODEL=$2
QG_MODEL=$3
BERT_PATH=$4

${PYT_QG} inference_e2e.py \
--fin=data/sharc/json/sharc_dev.json \
--dm=${DM_MODEL} \
--model_bert_base_path=${BERT_PATH} \
--model_recover_path=${QG_MODEL} \
--cache_path=pretrained_models \
--batch_size=4
