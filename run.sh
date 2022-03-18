#!/bin/bash
# To finetune and train sst2
#build directories
# cd directory
# ./build_task_dirs.sh bert sst2
# cd ..

# python ./run_files/run_glue_bert.py --TASK sst2 --MAX_LENGTH 64
# python ./tools/store_saliencies_glue.py --TASK sst2 --MAX_LENGTH 64
# python ./run_files/run_glue_w_lr.py --TASK sst2 --GAMMA 0.005 --PHI 0.0005 --SAVE_NAME run_1_sst2 --BATCH_SIZE 32 --DATA_SEED 42 --EPOCHS 5 --MAX_LENGTH 64 --LEARNING_RATE 3e-5

# To finetune and train ag_news
#build directories
cd directory
./build_task_dirs.sh bert ag_news 1
cd ..

python ./run_files/run_classification_bert.py --TASK ag_news
python ./tools/store_saliencies.py --TASK ag_news
python ./run_files/run_classification_w_lr.py --TASK ag_news --GAMMA 0.1 --PHI 0.1 --SAVE_NAME run_1_agnews --BATCH_SIZE 32 --DATA_SEED 42 --EPOCHS 5 --MAX_LENGTH 128 --LEARNING_RATE 3e-5
