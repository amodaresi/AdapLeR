import os
import sys
import pickle
import os.path as o

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

import argparse
from directories import *

### Parse inputs
parser = argparse.ArgumentParser()
parser.add_argument("--TASK")
parser.add_argument("--MAX_LENGTH", default=128, type=int)
parser.add_argument("--BATCH_SIZE", default=48, type=int)
parser.add_argument("--MULTI_GPU", action="store_true")
parser.add_argument("--LR_MODEL", action="store_true")
parser.add_argument("--GPU", default=0, type=int)
parser.add_argument("--MODEL_PATH")
args = parser.parse_args()

### Hyper Params
SELECTED_GPU = "multi" if args.MULTI_GPU else args.GPU
MAX_LENGTH = args.MAX_LENGTH
TASK = args.TASK
MODEL_PATH = 'bert-base-uncased' if MAX_LENGTH <= 512 else LONG_BERT_DIR
BATCH_SIZE = args.BATCH_SIZE
LOAD_MODEL_PATH = args.MODEL_PATH

# Import Requirements
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from transformers import (
    BertConfig,
    BertTokenizer, 
)

from modeling.modeling_tf_bert import TFBertForSequenceClassification
from utils.classification_utils import ModelCheckpoint_wlr
from utils.task_loaders import load_task

if SELECTED_GPU == "multi":
  mirrored_strategy = tf.distribute.MirroredStrategy()
else:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # tf.config.set_logical_device_configuration(
      #   gpus[int(SELECTED_GPU)],
      #   [tf.config.LogicalDeviceConfiguration(memory_limit=7100)])
      tf.config.experimental.set_visible_devices(gpus[int(SELECTED_GPU)], 'GPU')
      tf.config.experimental.set_memory_growth(gpus[int(SELECTED_GPU)], True)
      # tf.config.experimental.se.set_per_process_memory_fraction(0.92)
      print(gpus[int(SELECTED_GPU)])
    except RuntimeError as e:
      print(e)
  else:
    print('GPU not found!')


# Load Tokenizer & Dataset
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

### Load Task
datasets, info, metrics = load_task[TASK](tokenizer=tokenizer, max_length=MAX_LENGTH, training_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE)

num_labels = info["num_labels"]

config = BertConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)

if SELECTED_GPU == "multi":
    with mirrored_strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
        if args.LR_MODEL:
          model.to_AdapLeR()
          model.compile()
          model.load_weights(LOAD_MODEL_PATH)
        else:
          model.load_weights(LOAD_MODEL_PATH)
          model.to_AdapLeR()
          model.bert.encoder.ETA.assign(np.ones(12) * 1e-10)
else:
    model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
    if args.LR_MODEL:
      model.to_AdapLeR()
      model.compile()
      model.load_weights(LOAD_MODEL_PATH)
    else:
      model.load_weights(LOAD_MODEL_PATH)
      model.to_AdapLeR()
      model.bert.encoder.ETA.assign(np.ones(12) * 1e-10)

checkpoint = ModelCheckpoint_wlr(
          datasets["tests"], 
          metrics=metrics, 
          save_model=False, 
          saved_model_path="",
          return_flops_per_example=True  
)
checkpoint.set_model(model=model)
flops = checkpoint.on_epoch_end(1, inf_lambda=True)

with open(uppath(args.MODEL_PATH, 3) + "/logs/flops_per_example.pickle", "wb") as f:
  pickle.dump(flops, f, protocol=pickle.HIGHEST_PROTOCOL)