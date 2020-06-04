#!/usr/bin/env bash

function run(){
# run fine-tuning
export CUDA_VISIBLE_DEVICES=$1
BATCH_SIZE=$2  # 32
GRADACC=$3  # 8
LR=$4  # 0.00002
EPOCH=$5  # 10

MODEL='unilm'
OUTPUT_DIR=saved_models/"${MODEL}_${BATCH_SIZE}_${LR}_${EPOCH}"
MODEL_DIR=pretrained_models
MODEL_RECOVER_PATH=${MODEL_DIR}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/tmp/"${MODEL}_${BATCH_SIZE}_${LR}_${EPOCH}_cache"
mkdir -p ${OUTPUT_DIR}

DATA_DIR=data
FSRC='qg_train_src.txt'
FTGT='qg_train_tgt.txt'

PYT_QG=/path/to/emtqg/bin/python  # TODO
#PYT_QG=/research/king3/yfgao/miniconda3/envs/emtqg/bin/python

${PYT_QG} qg/biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids \
  --cache_path=${MODEL_DIR} \
  --data_dir ${DATA_DIR} --src_file ${FSRC} --tgt_file ${FTGT} \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 256 --max_position_embeddings 256 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GRADACC} \
  --learning_rate ${LR} --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs ${EPOCH}


TRAINED_MODEL_PATH="${OUTPUT_DIR}/model.${EPOCH}.bin"
EVAL_SPLIT=gold
FSRC='qg_dev_src.txt'
# run decoding
${PYT_QG} qg/biunilm/decode_seq2seq.py --bert_model bert-large-cased \
  --cache_path=${MODEL_DIR} \
  --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/${FSRC} --split ${EVAL_SPLIT} \
  --model_recover_path ${TRAINED_MODEL_PATH} \
  --max_seq_length 256 --max_tgt_length 48 \
  --batch_size 4 --beam_size 10 --length_penalty 0 \
  --output_file=${TRAINED_MODEL_PATH}

}


GPUID=$1
BS=16
GRADACC=4
LR=0.00002
EPOCH=20

run ${GPUID} ${BS} ${GRADACC} ${LR} ${EPOCH}
