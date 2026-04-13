export MASTER_PORT=29501
set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
OUTPUT_DIR="${DIR}/dataset_mydoc/checkpoint/v3_categoryv1.1_$(date +%F-%H)"
DATA_DIR="${DIR}/dataset_mydoc/dataset_category"

mkdir -p "${OUTPUT_DIR}"

deepspeed train.py \
  --model_dir ../layoutlmv3-base \
  --dataset_dir "${DATA_DIR}" \
  --dataloader_num_workers 1 \
  --deepspeed ds_config.json \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --do_train \
  --do_eval \
  --logging_steps 100 \
  --bf16 \
  --seed 42 \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --new_module_lr 1e-3 \
  --warmup_steps 100 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --remove_unused_columns False \
  --num_categories 24 \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  "$@"
