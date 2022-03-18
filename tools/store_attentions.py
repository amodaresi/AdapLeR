import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from directories import *
import argparse
import pickle

### Parse inputs
parser = argparse.ArgumentParser()

parser.add_argument("--TASK")
parser.add_argument("--SPLIT", default="train")
parser.add_argument("--MAX_LENGTH", default=128, type=int)
parser.add_argument("--BATCH_SIZE", default=4, type=int)
parser.add_argument("--ONLY_LAST_LAYER", action="store_true")
parser.add_argument("--ROLLOUT", action="store_true")
parser.add_argument("--DATA_SEED", default=42, type=int)
parser.add_argument("--EPOCH", default=5, type=int)
parser.add_argument("--GPU", default=0, type=int)
args = parser.parse_args()

# Hyper Params
BACKBONE = "bert"
SELECTED_GPU = args.GPU
TASK = args.TASK
SPLIT = args.SPLIT
ROLLOUT = args.ROLLOUT
ONLY_LAST_LAYER = args.ONLY_LAST_LAYER
MODEL_PATH = 'bert-base-uncased'
MAX_LENGTH = args.MAX_LENGTH
SAL_BATCH_SIZE = args.BATCH_SIZE
SEED = args.DATA_SEED
LOAD_MODEL_PATH = WORKER_MODEL_PATH(TASK, BACKBONE, seed=SEED)
EPOCH = args.EPOCH
SAL_AGG = "L2"
SAL_NORM = None
ATTENTION_SAVE_PATH = WORKER_ATTENTIONS_PATH_to_WRITE(TASK, BACKBONE, SPLIT, seed=SEED)

# Import Requirements
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from transformers import (
    BertConfig,
    BertTokenizer, 
    TFBertForSequenceClassification
)

from utils.task_loaders import load_task

def compute_joint_attention(att_mat):
    residual_att = np.eye(att_mat.shape[1])[None,...]
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    
    joint_attentions = np.zeros(aug_att_mat.shape)
    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions

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
      # tf.config.experimental.set_memory_growth(gpus[int(SELECTED_GPU)], True)
      # tf.config.experimental.se.set_per_process_memory_fraction(0.92)
      print(gpus[int(SELECTED_GPU)])
    except RuntimeError as e:
      print(e)
  else:
    print('GPU not found!')


# Load Tokenizer & Dataset
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

### Load Task
datasets, info, metrics = load_task[TASK](tokenizer=tokenizer, max_length=MAX_LENGTH, training_batch_size=SAL_BATCH_SIZE, eval_batch_size=SAL_BATCH_SIZE, seed=SEED)

num_steps = info[SPLIT + "_steps"]
num_labels = info["num_labels"]
num_examples = info[SPLIT + "_examples"]

if SPLIT == "val":
  first_key = list(datasets["evals"].keys())[0]
  selected_dataset = datasets["evals"][first_key]
elif SPLIT == "test":
  first_key = list(datasets["tests"].keys())[0]
  selected_dataset = datasets["tests"][first_key]
else:
  selected_dataset = datasets[SPLIT]
selected_dataset = selected_dataset.unbatch().repeat(-1).take((np.math.ceil(num_examples / SAL_BATCH_SIZE)) * SAL_BATCH_SIZE).batch(SAL_BATCH_SIZE)

# Load Model
config = BertConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
model.load_weights(LOAD_MODEL_PATH+str(EPOCH)+'.h5')

print("start to run")
attentions_mat = np.zeros((num_examples, MAX_LENGTH, MAX_LENGTH), dtype=np.float32) if ONLY_LAST_LAYER else np.ones((num_examples, 12, MAX_LENGTH, MAX_LENGTH), dtype=np.float32)
lengths = []
for ex, inputs in enumerate(selected_dataset):

    outputs = model(inputs[0], output_attentions=True)
    length = tf.reduce_sum(inputs[0]['attention_mask']).numpy()
    attn = tf.reduce_mean(tf.squeeze(tf.stack(outputs['attentions'])), axis=1)[:, :length, :length].numpy()
    if ROLLOUT:
      attn = compute_joint_attention(attn)
    
    if ONLY_LAST_LAYER:
      attentions_mat[ex, :length, :length] = attn[-1]
    else:
      attentions_mat[ex, :, :length, :length] = attn

    lengths.append(length)

print("start to save")
name = "attn_rollout" if ROLLOUT else "attn"
if ONLY_LAST_LAYER:
  print("save pickle")
  name = name + "_last"

if ONLY_LAST_LAYER:
  attentions_list = []
  for ex in range(num_examples):
    attentions_list.append(attentions_mat[ex, :lengths[ex], :lengths[ex]] if ONLY_LAST_LAYER else attentions_mat[ex, :, :lengths[ex], :lengths[ex]])
  del attentions_mat
  with open(ATTENTION_SAVE_PATH+f"{name}_wCLSSEP_{EPOCH}.pickle", 'wb') as f:
    pickle.dump(attentions_list, f)
else:
  print("save numpy")
  with open(ATTENTION_SAVE_PATH+f"{name}_wCLSSEP_{EPOCH}.npy", 'wb') as f:
    np.save(f, attentions_mat)

print("finished")