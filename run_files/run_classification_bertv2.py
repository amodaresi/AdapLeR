import argparse
import logging
import json
import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
from directories import *

### Parse inputs
parser = argparse.ArgumentParser()

parser.add_argument("--TASK")
parser.add_argument("--MAX_LENGTH", default=128, type=int)
parser.add_argument("--BATCH_SIZE", default=32, type=int)
parser.add_argument("--EVAL_BATCH_SIZE", default=32, type=int)
parser.add_argument("--ACC_STEPS", default=1, type=int)
parser.add_argument("--LEARNING_RATE", default=2e-5, type=float)
parser.add_argument("--WARMUP_RATIO", default=0.06, type=float)
parser.add_argument("--MULTI_GPU", action="store_true")
parser.add_argument("--EPOCHS", default=5, type=int)
parser.add_argument("--DATA_SEED", default=42, type=int)
parser.add_argument("--GPU", default=0, type=int)
args = parser.parse_args()

### Hyper Params
SELECTED_GPU = "multi" if args.MULTI_GPU else args.GPU
TASK = args.TASK
MAX_LENGTH = args.MAX_LENGTH
MODEL_PATH = 'bert-base-uncased' if MAX_LENGTH <= 512 else LONG_BERT_DIR
BATCH_SIZE = args.BATCH_SIZE
EVAL_BATCH_SIZE = args.EVAL_BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
WARMUP_RATIO = args.WARMUP_RATIO
WEIGHT_DECAY_RATE = 0.01
EPOCHS = args.EPOCHS
SAVE_MODEL = True
SEED = args.DATA_SEED
SAVED_MODEL_PATH = f"./directory/bert/{SEED}/{TASK}/models/bert_"+TASK
ACCUMULATION_STEPS = args.ACC_STEPS

### Import Requirements
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm 


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from transformers import (
    BertConfig,
    BertTokenizer,
    create_optimizer
)

from modeling.modeling_tf_bert_Acc import TFBertForSequenceClassificationWAcc

if SELECTED_GPU == 'multi':
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
else:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.set_logical_device_configuration(
        gpus[int(SELECTED_GPU)],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7200)])
      tf.config.experimental.set_visible_devices(gpus[int(SELECTED_GPU)], 'GPU')
      # tf.config.experimental.set_memory_growth(gpus[int(SELECTED_GPU)], True)
      # tf.config.experimental.se.set_per_process_memory_fraction(0.92)
      print(gpus[int(SELECTED_GPU)])
    except RuntimeError as e:
      print(e)
  else:
    print('GPU not found!')

from utils.classification_utils import ModelCheckpoint
from utils.task_loaders import load_task

### Load Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


### Load Task
datasets, info, metrics = load_task[TASK](tokenizer=tokenizer, max_length=MAX_LENGTH, training_batch_size=np.ceil(BATCH_SIZE / ACCUMULATION_STEPS).astype(np.int32), eval_batch_size=EVAL_BATCH_SIZE, seed=SEED)

train_steps = info["train_steps"]
num_labels = info["num_labels"]
if "class_weights" in info:
  class_weights = info["class_weights"]
else:
  class_weights = None

### Load Model
print("loading model")
opt, scheduler = create_optimizer(init_lr=LEARNING_RATE,
                                  num_train_steps=train_steps * EPOCHS,
                                  num_warmup_steps=WARMUP_RATIO * train_steps * EPOCHS,
                                  adam_epsilon = 1e-8,
                                  weight_decay_rate = WEIGHT_DECAY_RATE)

   

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

config = BertConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)
if SELECTED_GPU == 'multi':
  with strategy.scope():
    model = TFBertForSequenceClassificationWAcc.from_pretrained(MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
    model.compile(optimizer=opt, loss=loss, accum_steps=ACCUMULATION_STEPS, class_weights=class_weights)
else:
  model = TFBertForSequenceClassificationWAcc.from_pretrained(MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
  model.compile(optimizer=opt, loss=loss, accum_steps=ACCUMULATION_STEPS, class_weights=class_weights)

checkpoint = ModelCheckpoint(
  datasets["evals"], 
  metrics=metrics, 
  save_model=SAVE_MODEL, 
  saved_model_path=SAVED_MODEL_PATH)

print("starting training")
history = model.fit(datasets["train"].repeat(),
                    epochs=EPOCHS,
                    steps_per_epoch=train_steps,
                    callbacks=[checkpoint]
                    )

with open(f"./directory/bert/{SEED}/{TASK}/logs/ft_logs.json", "w", encoding='utf-8') as jsonfile:
  json.dump(checkpoint.all_scores,jsonfile,ensure_ascii=False)