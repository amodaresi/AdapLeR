import sys
import os.path as o

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from directories import *
import argparse
import logging

### Parse inputs
parser = argparse.ArgumentParser()
parser.add_argument("--GAMMA", default=1.0, type=float)
parser.add_argument("--PHI", default=10.0, type=float)
parser.add_argument("--GPU", default=0, type=int)
parser.add_argument("--EPOCHS", default=5, type=int)
parser.add_argument("--MULTI_GPU", action="store_true")
parser.add_argument("--TASK")
parser.add_argument("--INITIAL_MODEL_PATH", default="pretrain")
parser.add_argument("--MAX_LENGTH", default=128, type=int)
parser.add_argument("--DATA_SEED", default=42, type=int)
parser.add_argument("--SAVE_NAME", default="znorm_l2")
parser.add_argument("--WARMUP_RATIO", default=0.06, type=float)
parser.add_argument("--EVAL_BATCH_SIZE", default=32, type=int)
parser.add_argument("--LEARNING_RATE", default=3e-5, type=float)
parser.add_argument("--ACC_STEPS", default=1, type=int)
parser.add_argument("--BATCH_SIZE", default=32, type=int)
args = parser.parse_args()

# Hyper Params
SELECTED_GPU = "multi" if args.MULTI_GPU else args.GPU
TASK = args.TASK
MAX_LENGTH = args.MAX_LENGTH
BASE_MODEL_PATH = 'bert-base-uncased' if MAX_LENGTH <= 512 else LONG_BERT_DIR
INITIAL_MODEL_PATH = args.INITIAL_MODEL_PATH
TRAINING_BATCH_SIZE = args.BATCH_SIZE
EVAL_BATCH_SIZE = 32
ACCUMULATION_STEPS = args.ACC_STEPS
LEARNING_RATE = args.LEARNING_RATE
EXP_LEARNING_RATE = 0.001
WARMUP_RATIO = args.WARMUP_RATIO
WEIGHT_DECAY_RATE = 0.01
EPOCHS = args.EPOCHS
SAVE_MODEL = True 
SEED = args.DATA_SEED
SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ(args.TASK, "bert", seed=SEED)
LOG_PATH = WORKER_LOG_SAVE_PATH(TASK, "bert", 
args.SAVE_NAME, seed=SEED, GAMMA=args.GAMMA, PHI=args.PHI)
SAVED_MODEL_PATH = WORKER_MODEL_SAVE_PATH(TASK, "bert", 
args.SAVE_NAME, seed=SEED, GAMMA=args.GAMMA, PHI=args.PHI)

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode="w")        
    handler.setFormatter(logging.Formatter('%(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
    
logger = setup_logger("run_glue_logger", LOG_PATH+".log")
logger.info("Input args: %r", args)
print(args)

# Import Requirements
import random
import numpy as np
import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tqdm import tqdm

from datasets import load_dataset

from transformers import (
    BertConfig,
    BertTokenizer, 
    create_optimizer,
)

from modeling.modeling_tf_bert import TFBertForSequenceClassification
from utils.task_loaders import load_task, get_best_epochs
from utils.classification_utils import ModelCheckpoint_wlr

mirrored_strategy = None
if SELECTED_GPU == "multi":
  mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
else:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.set_logical_device_configuration(
        gpus[int(SELECTED_GPU)],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7100)])
      tf.config.experimental.set_visible_devices(gpus[int(SELECTED_GPU)], 'GPU')
      # tf.config.experimental.set_memory_growth(gpus[int(SELECTED_GPU)], True)
      print(gpus[int(SELECTED_GPU)])
    except RuntimeError as e:
      print(e)
  else:
    print('GPU not found!')

### Load Tokenizer 
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_PATH)

### Load Task
datasets, info, metrics, sals = load_task[TASK](tokenizer=tokenizer, max_length=MAX_LENGTH, training_batch_size=TRAINING_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE, seed=SEED, include_sals=True)

train_steps = info["train_steps"]
num_labels = info["num_labels"]
if "class_weights" in info:
  class_weights = info["class_weights"]
else:
  class_weights = None

if SELECTED_GPU == "multi":
  datasets["train"] = mirrored_strategy.experimental_distribute_dataset(datasets["train"])
  for k in datasets["evals"].keys():
      datasets["evals"][k] = mirrored_strategy.experimental_distribute_dataset(datasets["evals"][k])

  sals = mirrored_strategy.experimental_distribute_dataset(sals)

config = BertConfig.from_pretrained(BASE_MODEL_PATH, num_labels=num_labels)

def load_model():
  global INITIAL_MODEL_PATH
  model = TFBertForSequenceClassification.from_pretrained(BASE_MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
  if INITIAL_MODEL_PATH == "":
    best_epoch = get_best_epochs(f"./directory/bert/{SEED}/{TASK}/logs/ft_logs.json")[-1] + 1
    INITIAL_MODEL_PATH = f"./directory/bert/{SEED}/{TASK}/models/bert_{TASK}_{best_epoch}.h5"
  if INITIAL_MODEL_PATH != "pretrain":
    model.load_weights(INITIAL_MODEL_PATH)
  model.to_AdapLeR()
  return model

def optimizers():
  opt, _ = create_optimizer(init_lr=LEARNING_RATE,
                                  num_train_steps=train_steps * (EPOCHS),
                                  num_warmup_steps=WARMUP_RATIO * train_steps * EPOCHS,
                                  adam_epsilon = 1e-12,
                                  weight_decay_rate = WEIGHT_DECAY_RATE)

  zeta_opt, _ = create_optimizer(init_lr=5e-4,
                                    num_train_steps=train_steps * EPOCHS,
                                    num_warmup_steps=WARMUP_RATIO * train_steps * EPOCHS,
                                    adam_epsilon = 1e-12)

  opt.global_clipnorm = 1
  opt._global_clipnorm = 1

  zeta_opt.global_clipnorm = 1
  zeta_opt._global_clipnorm = 1
  return opt, zeta_opt

def compile(model, opt, zeta_opt):
  model.compile(optimizer=opt,
                speedup_optimizer=zeta_opt,
                GAMMA=args.GAMMA,
                PHI=args.PHI,
                accum_steps=ACCUMULATION_STEPS,
                class_weights=class_weights
              )

# Load Model
if SELECTED_GPU == "multi":
  with mirrored_strategy.scope():
    model = load_model()
    opt, zeta_opt = optimizers()
    compile(model, opt, zeta_opt)
else:
  model = load_model()
  opt, zeta_opt = optimizers()
  compile(model, opt, zeta_opt)

@tf.function
def train(inputs, sal_inputs, strategy=None):
    if strategy:
        per_replica_outputs = strategy.run(model.train_phase_1, args=(inputs, sal_inputs))

        kl_losses = tf.squeeze(strategy.reduce("MEAN", per_replica_outputs["kl_head_losses"], axis=0))
        loss = strategy.reduce("MEAN", per_replica_outputs["loss"], axis=None)
        combined_loss = strategy.reduce("MEAN", per_replica_outputs["combined_loss"], axis=None)
        drops = strategy.reduce("MEAN", per_replica_outputs["drops"], axis=None)
        length_loss = strategy.reduce("MEAN", per_replica_outputs["length_loss"], axis=None)
        total_loss = strategy.reduce("MEAN", per_replica_outputs["total_loss"], axis=None)
        head_loss = strategy.reduce("MEAN", per_replica_outputs["head_loss"], axis=None)
        _lambda = strategy.reduce("MEAN", model.bert.encoder._lambda, axis=None)
        _gamma = mirrored_strategy.reduce("MEAN", per_replica_outputs["gamma"], axis=None)
    else:
        outputs = model.train_phase_1(inputs, sal_inputs)
        kl_losses = tf.reduce_mean(outputs["kl_head_losses"], axis=0)
        drops = outputs["drops"]
        loss = outputs["loss"]
        total_loss = outputs["total_loss"]
        head_loss = outputs["head_loss"]
        combined_loss = outputs["combined_loss"]
        length_loss = outputs["length_loss"]
        _lambda = model.bert.encoder._lambda
        _gamma = outputs["gamma"]

    return {
        "total_loss": total_loss,
        "head_loss": head_loss,
        "combined_loss": combined_loss,
        "length_loss": length_loss,
        "loss": loss,
        "drops": drops,
        "lambda": _lambda,
        "gamma": _gamma,
        "kl_losses": kl_losses
    }


train_signals = {
  "loss": [],
  "head_loss": [],
  "lambda": [],
  "gamma": []
}

print(SAVED_MODEL_PATH)
avg_total_loss = 0
min_window = []
ratios_window = []

checkpoint = ModelCheckpoint_wlr(
    datasets["evals"], 
    metrics=metrics, 
    save_model=SAVE_MODEL, 
    saved_model_path=SAVED_MODEL_PATH,
    logger=logger,
    strategy=mirrored_strategy,
    best_score=np.max(get_best_epochs(f"./directory/bert/{SEED}/{TASK}/logs/ft_logs.json", return_scores=True))
)
checkpoint.set_model(model=model)

prev_slope = 0
for epoch in range(EPOCHS):
  sal_mat_iter = iter(sals)
  print("Epoch {}/{}".format(epoch+1, EPOCHS))
  logger.info(f"\nEpoch: {epoch+1}")
  pbar = tqdm(enumerate(datasets["train"]), total=train_steps, leave=True)
  avg_loss = 0
  avg_head_loss = 0
  avg_length_loss = 0
  avg_combined_loss = 0
  avg_drops = 0
  for step, inputs in pbar:
    sal_mat = sal_mat_iter.next()
    outputs = train(
        inputs, 
        sal_inputs=sal_mat,
        strategy=mirrored_strategy
        )
    avg_loss = (avg_loss * step + outputs['loss'].numpy()) / (step + 1)
    avg_head_loss = (avg_head_loss * step + outputs['head_loss'].numpy()) / (step + 1)
    avg_total_loss = (avg_total_loss * (step + epoch * train_steps) + outputs['total_loss'].numpy()) / (step + epoch * train_steps + 1)
    avg_length_loss = (avg_length_loss * step + outputs['length_loss'].numpy()) / (step + 1)
    avg_combined_loss = (avg_combined_loss * step + outputs['combined_loss'].numpy()) / (step + 1)
    avg_drops = (avg_drops * step + outputs['drops'].numpy()) / (step + 1)
    pbar.set_postfix({"rt_loss": outputs['loss'].numpy(), 'loss': avg_loss, "re_hloss": outputs['head_loss'].numpy(), 'head_loss': avg_head_loss, "re_lloss": outputs['length_loss'].numpy(), "length_loss": avg_length_loss, "re_comb": outputs['combined_loss'].numpy(), "comb_loss": avg_combined_loss, "re_drops": outputs['drops'].numpy(), "drops": avg_drops, "lambda": outputs["lambda"].numpy()})
    for k in train_signals.keys():
      train_signals[k].append(outputs[k].numpy())
    
    nan_found = False
    if np.isnan(outputs["loss"].numpy()) or np.isnan(outputs["head_loss"].numpy()):
      nan_found = True
    if nan_found:
      print(outputs["kl_losses"])
      print(model.bert.encoder.ETA.numpy())
      print(model.bert.encoder.THETA.numpy())

    if nan_found:
      print("NAN BREAK")
      exit()

    if EPOCHS == 10:
      model.bert.encoder._lambda.assign(10.0 * 3.0 ** (epoch))
    elif EPOCHS == 3:
      model.bert.encoder._lambda.assign(50.0 * 50.0 ** (epoch))
    else:
      model.bert.encoder._lambda.assign(10.0 * 10.0 ** (epoch))

    if step % 100 == 0:
      print("ZETAs: \n" , model.bert.encoder.ETA.numpy(), flush=True)
      print("THETAs: \n" , model.bert.encoder.THETA.numpy(), flush=True)
      print("KL_LOSSES: \n" , outputs["kl_losses"], flush=True)
      

  print()
  checkpoint.on_epoch_end(epoch, inf_lambda=True)


import pickle

with open(LOG_PATH + '.pkl', 'wb') as f:
    pickle.dump(train_signals, f, pickle.HIGHEST_PROTOCOL)

logger.info("completed!")