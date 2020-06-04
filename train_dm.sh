#!/usr/bin/env bash

function train(){
    PYT_EMT=/path/to/emt/bin/python  # TODO
#    PYT_EMT=/research/king3/yfgao/miniconda3/envs/emt/bin/python
    export CUDA_VISIBLE_DEVICES=$1
    BERTMODELDIR=pretrained_models/bert-base-uncased.tar.gz
    DATA=entail_bu
    LOSS_ENTAIL_WEIGHT=10
    LOSS_SPAN_WEIGHT=0.6
    TRAIN_BATCH=10
    GRADACC=1
    EPOCH=5
    LEARNING_RATE=5e-5
    MODEL='c2f_entail'
    SEED=28

    SAVE_DIR="saved_models/lew_${LOSS_ENTAIL_WEIGHT}_lsw_${LOSS_SPAN_WEIGHT}"
    mkdir -p ${SAVE_DIR}
    PREFIX="seed_${SEED}"
    mkdir -p "${SAVE_DIR}/${PREFIX}"

    ${PYT_EMT} -u train_dm.py \
    --train_batch=${TRAIN_BATCH} \
    --gradient_accumulation_steps=${GRADACC} \
    --epoch=${EPOCH} \
    --seed=${SEED} \
    --learning_rate=${LEARNING_RATE} \
    --loss_span_weight=${LOSS_SPAN_WEIGHT} \
    --loss_entail_weight=${LOSS_ENTAIL_WEIGHT} \
    --dsave="${SAVE_DIR}/{}" \
    --model=${MODEL} \
    --early_stop=dev_combined \
    --data=data/ \
    --data_type=${DATA} \
    --prefix=${PREFIX} \
    --eval_every_steps=500 \
    --bert_model_path=${BERTMODELDIR}
}

GPUID=$1

train ${GPUID}
