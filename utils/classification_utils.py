import os
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import sent_tokenize

def text_examples_to_tfdataset(examples, 
                               tokenizer,
                               max_length=128,
                               text_keyname="text",
                               label_keyname="label"
                               ):
    
    def gen():
        for example in examples:
          inputs = tokenizer.encode_plus(example[text_keyname], max_length=max_length, truncation=True, padding="max_length" if max_length else "do_not_pad")

          label = example[label_keyname]

          yield ({'input_ids': inputs['input_ids'],
                 'token_type_ids': inputs['token_type_ids'],
                 'attention_mask': inputs['attention_mask']},
                 label)

    label_type = tf.int32

    return tf.data.Dataset.from_generator(gen,
        ({'input_ids': tf.int32,
         'token_type_ids': tf.int32,
         'attention_mask': tf.int32},
         label_type),
        ({'input_ids': tf.TensorShape([None]),
         'token_type_ids': tf.TensorShape([None]),
         'attention_mask': tf.TensorShape([None])},
         tf.TensorShape([])))

# convertors = {
#     "ag_news": text_examples_to_tfdataset,
#     "imdb": text_examples_to_tfdataset,
#     "dbpedia_14": content_examples_to_tfdataset,
# }


def calc_flops_per_length_bert(n, num_labels):
  classifier_flops = 1537
  c = 1183490 - classifier_flops
  b = 170216500
  a = 37872
  return a*(n**2) + b*n + c + num_labels * classifier_flops


def calc_flops_per_length_bert_w_lr(layers_n, num_labels):
  def layer_flop_calculator(n):
    flops = 3156 * n**2 + 14233554 * n
    return flops
  
  classifier_flops = 1537
  flops = 1183537 + (num_labels - 1) * classifier_flops + 6152 * layers_n[0]
  for n in layers_n:
    flops += layer_flop_calculator(n)
  
  return flops

# Callback Full Checkpoint
class ModelCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, datasets, metrics, save_model, saved_model_path):
    super(ModelCheckpoint, self).__init__()
    self.metrics = metrics
    self.save_model = save_model
    self.saved_model_path = saved_model_path
    self.datasets: dict = datasets
    self.eval_labels: np.ndarray = None
    self.test_labels: np.ndarray = None
    self.all_scores = []

  def evaluate(self):
    scores = {}
    for split_name, dataset in self.datasets.items():
      y_preds = []
      y_true = []
      for batch in dataset:
        output = self.model(batch[0])
        y_preds.extend(tf.math.argmax(output.logits, axis=1).numpy())
        y_true.extend(batch[1].numpy())

      scores[split_name] = {}
      for metric_name, metric_func in self.metrics.items():
        scores[split_name][metric_name] = metric_func(y_true, y_preds)
    return scores

  def on_epoch_end(self, epoch, logs=None):
    scores = self.evaluate()
    for split_name, split_scores in scores.items():
      str_result = split_name + ": "
      for metric_name, metric_value in split_scores.items():
        str_result = str_result + " - {0}: {1:.4f}".format(metric_name, metric_value)
      print(str_result)

    if self.save_model:
      self.model.save_weights(self.saved_model_path+"_"+str(epoch+1)+".h5") 
    
    self.all_scores.append(scores)


# Callback Full Checkpoint
class ModelCheckpoint_wlr(tf.keras.callbacks.Callback):
  def __init__(self, datasets, metrics, save_model, saved_model_path, logger=None, strategy=None, best_score=None, THR=0.01, return_flops_per_example=False):
    super(ModelCheckpoint_wlr, self).__init__()
    self.metrics = metrics
    self.save_model = save_model
    self.saved_model_path = saved_model_path
    self.datasets: dict = datasets
    self.eval_labels: np.ndarray = None
    self.test_labels: np.ndarray = None
    self.strategy = strategy
    self.logger = logger
    self.best_score = best_score
    self.return_flops_per_example = return_flops_per_example
    if best_score:
      self.best_speedup = 0.0
      self.score_threshold = best_score - THR
      print("Score threshold:", self.score_threshold)
      self.logger.info(f"Score threshold: {self.score_threshold}")


  @tf.function
  def eval_wo_sals(self, inputs, labels):
    outputs = self.model(inputs, lr_mode=False, output_explanations=True)
    y_preds = tf.argmax(outputs.logits, axis=-1, output_type=tf.int32)
    return (y_preds, tf.transpose(tf.stack(outputs.explanations), perm=[1, 0, 2]), outputs.logits)

  @tf.function
  def distributed_eval(self, inputs):
    per_replica_outputs = self.strategy.run(self.eval_wo_sals, args=(inputs))
    return (
      self.strategy.gather(per_replica_outputs[0], axis=0),
      self.strategy.gather(per_replica_outputs[1], axis=0),
      self.strategy.gather(per_replica_outputs[2], axis=0),
      self.strategy.gather(inputs[1], axis=0)
    )

  def evaluate(self, inf_lambda=True):
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    if inf_lambda:
      _lambda = self.model.bert.encoder._lambda.numpy()
      self.model.bert.encoder._lambda.assign(1e+20)

    scores = {}
    for split_name, dataset in self.datasets.items():
      y_preds = []
      losses = []
      y_true = []
      lengths = np.empty((0, 13), dtype=np.int32)
      for batch in dataset:
        if self.strategy is not None:
          outputs = self.distributed_eval(batch)
          preds = outputs[0]
          exps = outputs[1]
          logits = outputs[2]
          labels = outputs[3]
        else:
          outputs = self.eval_wo_sals(batch[0], batch[1])
          preds = outputs[0]
          exps = outputs[1]
          logits = outputs[2]
          labels = batch[1]
        y_preds.extend(preds.numpy())
        y_true.extend(labels.numpy())
        # losses.extend(loss_fn(labels, logits).numpy())
        last_layer_length = np.sum(exps[:, -1] > self.model.bert.encoder.ETA[-1], axis=-1)
        lengths = np.concatenate([lengths, np.concatenate([np.sum(exps > 1e-6, axis=2), np.expand_dims(last_layer_length, axis=-1)], axis=1)])

      scores[split_name] = {}
      first_metric = None
      for metric_name, metric_func in self.metrics.items():
        if first_metric is None:
          first_metric = metric_name
        scores[split_name][metric_name] = metric_func(y_true, y_preds)
      
      if self.return_flops_per_example:
        flops_bert_arr = []
        flops_lr_arr = []
      else:
        flops_bert = 0
        flops_lr = 0

      for l in lengths:
        f_bert = calc_flops_per_length_bert(l[0], np.max(y_preds) + 1)
        f_adapler = calc_flops_per_length_bert_w_lr(l[1:], np.max(y_preds) + 1)
        if self.return_flops_per_example:
          flops_bert_arr.append(f_bert)
          flops_lr_arr.append(f_adapler)
        else:
          flops_bert += f_bert
          flops_lr += f_adapler
      if self.return_flops_per_example:
        flops_bert = np.sum(flops_bert_arr)
        flops_lr = np.sum(flops_lr_arr)
      scores[split_name]["COUNT"] = len(lengths)
      scores[split_name]["SPEEDUP (TOTAL)"] = flops_bert / flops_lr
        
    if inf_lambda:
      self.model.bert.encoder._lambda.assign(_lambda)
    if self.return_flops_per_example:
      return scores, first_metric, {"bert":flops_bert_arr, "adapler":flops_lr_arr}
    return scores, first_metric

  def predict(self, inf_lambda=True):
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    if inf_lambda:
      _lambda = self.model.bert.encoder._lambda.numpy()
      self.model.bert.encoder._lambda.assign(1e+20)

    return_outputs = {}
    for split_name, dataset in self.datasets.items():
      y_preds = []
      lengths = np.empty((0, 13), dtype=np.int32)
      for batch in dataset:
        if self.strategy is not None:
          outputs = self.distributed_eval(batch)
          preds = outputs[0]
          exps = outputs[1]
        else:
          outputs = self.eval_wo_sals(batch[0], batch[1])
          preds = outputs[0]
          exps = outputs[1]
        y_preds.extend(preds.numpy())
        last_layer_length = np.sum(exps[:, -1] > self.model.bert.encoder.ETA[-1], axis=-1)
        lengths = np.concatenate([lengths, np.concatenate([np.sum(exps > 1e-6, axis=2), np.expand_dims(last_layer_length, axis=-1)], axis=1)])

      return_outputs[split_name] = {"preds": y_preds}
      
      if self.return_flops_per_example:
        flops_bert_arr = []
        flops_lr_arr = []
      else:
        flops_bert = 0
        flops_lr = 0

      for l in lengths:
        f_bert = calc_flops_per_length_bert(l[0], np.max(y_preds) + 1)
        f_adapler = calc_flops_per_length_bert_w_lr(l[1:], np.max(y_preds) + 1)
        if self.return_flops_per_example:
          flops_bert_arr.append(f_bert)
          flops_lr_arr.append(f_adapler)
        else:
          flops_bert += f_bert
          flops_lr += f_adapler
      if self.return_flops_per_example:
        flops_bert = np.sum(flops_bert_arr)
        flops_lr = np.sum(flops_lr_arr)
      return_outputs[split_name]["COUNT"] = len(lengths)
      return_outputs[split_name]["SPEEDUP (TOTAL)"] = flops_bert / flops_lr
        
    if inf_lambda:
      self.model.bert.encoder._lambda.assign(_lambda)
    if self.return_flops_per_example:
      return return_outputs, {"bert":flops_bert_arr, "adapler":flops_lr_arr}
    return return_outputs

  def on_epoch_end(self, epoch, logs=None, inf_lambda=False):
    if self.return_flops_per_example:
      scores, first_metric, flops = self.evaluate(inf_lambda=inf_lambda)
    else:
      scores, first_metric = self.evaluate(inf_lambda=inf_lambda)
    for split_name, split_scores in scores.items():
      str_result = split_name + ": "
      for metric_name, metric_value in split_scores.items():
        str_result = str_result + " - {0}: {1:.4f}".format(metric_name, metric_value)
      print(str_result, flush=True)
      if self.logger:
        self.logger.info(str_result)
    
    print(self.model.bert.encoder.ETA.numpy())
    if self.logger:
      self.logger.info(f"Confidence Ratios: {self.model.bert.encoder.ETA.numpy()}")

    if self.best_score:
      dev_speedup = scores[list(scores.keys())[0]]["SPEEDUP (TOTAL)"]
      dev_score = scores[list(scores.keys())[0]][first_metric]
      if dev_speedup > self.best_speedup and inf_lambda and self.score_threshold < dev_score:
          print("Speedup improved from:", self.best_speedup, dev_speedup, flush=True)
          self.logger.info(f"Speedup improved from: {self.best_speedup}, {dev_speedup}")
          self.best_speedup = dev_speedup

          if self.save_model:
            self.model.save_weights(self.saved_model_path+"_lr.h5") 

    if self.return_flops_per_example:
      return flops
