#!/bin/bash
set -euo pipefail

date

# ===============================
# Custom single-dataset training
# ===============================
# Expected dataset layout:
#   /path/to/your_dataset/
#     image/
#       xxx.jpg
#       yyy.jpg
#     coco/
#       xxx.json
#       yyy.json
#     list_train.txt  # lines can be image filenames or coco filenames
#     list_val.txt
#
# Images are read from `images[0].file_name` in each COCO file.
# Relative file_name values are resolved from DATASET_ROOT.
# COCO files are loaded from COCO_ROOT/coco/*.json (can differ from DATASET_ROOT).
# If you only have one split list (list.txt), duplicate it as list_train.txt
# and list_val.txt for a quick smoke run.

DATASET_ROOT=/path/to/your_dataset
COCO_ROOT=/path/to/your_coco_root

TRAIN_PATH_1=${DATASET_ROOT}
EVAL_PATH_1=${DATASET_ROOT}
TRAIN_COCO_PATH_1=${COCO_ROOT}
EVAL_COCO_PATH_1=${COCO_ROOT}

MODEL_SIZE=base
SAVE_PATH=./outputs/outputs_train_custom/
MAX_NUM=128

SHORT_RANGE=896,1088
PATCH_SIZE=768,768
PATCH_NUM=1
KEEP_SIZE=False

BATCH_SIZE=2
LEARNING_RATE=2e-5
MOMENTUM=0.9
WEIGHT_DECAY=1e-2
LR_SCHEDULER=cosine

FINE_TUNE=True
RESTORE_FROM=./pretrained_model/docsam_base_all_dataset.pth
SNAPSHOT_DIR=./snapshots/custom_textregions_textlines/
START_ITER=0
TOTAL_ITER=12000
GPU_IDS=0

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 train.py \
  --train-path ${TRAIN_PATH_1} \
  --eval-path ${EVAL_PATH_1} \
  --train-coco-path ${TRAIN_COCO_PATH_1} \
  --eval-coco-path ${EVAL_COCO_PATH_1} \
  --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM} \
  --short-range ${SHORT_RANGE} --patch-size ${PATCH_SIZE} --patch-num ${PATCH_NUM} --keep-size ${KEEP_SIZE} \
  --batch-size ${BATCH_SIZE} --learning-rate ${LEARNING_RATE} --momentum ${MOMENTUM} --weight-decay ${WEIGHT_DECAY} --lr-scheduler ${LR_SCHEDULER} \
  --fine-tune ${FINE_TUNE} --restore-from ${RESTORE_FROM} --snapshot-dir ${SNAPSHOT_DIR} \
  --start-iter ${START_ITER} --total-iter ${TOTAL_ITER} --gpus ${GPU_IDS}
