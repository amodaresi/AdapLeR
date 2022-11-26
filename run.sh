#!/bin/bash
# To finetune and train sst2
# build directories
cd directory
./build_task_dirs.sh bert sst2 22
cd ..

python ./run_files/run_classification_bert.py --TASK sst2 --MAX_LENGTH 64 --DATA_SEED 22
python ./tools/store_saliencies.py --TASK sst2 --DATA_SEED 22 --MAX_LENGTH 64 --BATCH_SIZE 4
python ./run_files/run_classification_w_lr.py --TASK sst2 --GAMMA 0.005 --PHI 0.0005 --SAVE_NAME run_1_sst2 --BATCH_SIZE 32 --DATA_SEED 22 --EPOCHS 5 --MAX_LENGTH 64 --LEARNING_RATE 3e-5
python ./run_files/run_glue_prediction.py --TASK sst2 --MAX_LENGTH 64 --LR_MODEL --MODEL_PATH PATH_TO_MODEL.h5 --LR_MODEL
