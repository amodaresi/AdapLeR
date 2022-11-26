import os
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst2": 2,
    "stsb": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

glue_tasks_inputs = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

glue_tasks_metrics = {
  "cola": {'matthews': matthews_corrcoef},
  "mnli": {'accuracy': accuracy_score},
  "mrpc": {'accuracy': accuracy_score, 'f1': f1_score},
  "sst2": {'accuracy': accuracy_score},
  "stsb": {'spearman': spearmanr, 'pearson': pearsonr},
  "qqp": {'accuracy': accuracy_score, 'f1': f1_score},
  "qnli": {'accuracy': accuracy_score},
  "rte": {'accuracy': accuracy_score},
  "wnli": {'accuracy': accuracy_score},
}

def glue_examples_to_tfdataset(examples, 
                               tokenizer,
                               task,
                               max_length=128,
                               ):
    
    def gen():
        for example in examples:
          sentence1_key, sentence2_key = glue_tasks_inputs[task]
          args = (
                (example[sentence1_key],) if sentence2_key is None else (
                    example[sentence1_key], example[sentence2_key])
                )
          inputs = tokenizer.encode_plus(*args, max_length=max_length, truncation=True, padding="max_length" if max_length else "do_not_pad")

          label = example['label']

          yield ({'input_ids': inputs['input_ids'],
                 'token_type_ids': inputs['token_type_ids'],
                 'attention_mask': inputs['attention_mask']},
                 label)

    label_type = tf.float32 if task == "sts-b" else tf.int32

    return tf.data.Dataset.from_generator(gen,
        ({'input_ids': tf.int32,
         'token_type_ids': tf.int32,
         'attention_mask': tf.int32},
         label_type),
        ({'input_ids': tf.TensorShape([None]),
         'token_type_ids': tf.TensorShape([None]),
         'attention_mask': tf.TensorShape([None])},
         tf.TensorShape([])))

def save_pred_glue(predict, TASK=None, OUTPUT_DIR=None, NAME_ADD=""):
    indexes = np.arange(len(predict))

    if TASK == 'mrpc' or TASK == 'cola' or TASK == 'sst2' or TASK == 'qqp':
        data_labels = {0:'0', 1:'1'}
    elif TASK == 'mnli':
        data_labels = {0:'entailment', 1:'neutral', 2:'contradiction'}
    elif TASK == 'qnli' or TASK == 'rte':
        data_labels = {0:'entailment', 1:'not_entailment'}
    elif TASK == 'sts-b':
        data_labels = {}
    else:
        raise ValueError('No such TASK available.')

    ## Write the predictions in file
    TEST_PRED_FILE = os.path.join(OUTPUT_DIR, f"glue_prediction_{NAME_ADD}.tsv")

    if TASK == "sts-b":
        with open(TEST_PRED_FILE, 'w') as test_fh:
            test_fh.write("identity\tlabel\n")
            for idx in range(indexes.shape[0]):
                test_fh.write("%s\t%s\n" % (indexes[idx], predict[idx][0]))
    else:
        with open(TEST_PRED_FILE, 'w') as test_fh:
            test_fh.write("index\tprediction\n")
            for idx in range(indexes.shape[0]):
                test_fh.write("%s\t%s\n" % (indexes[idx], data_labels[predict[idx]]))
# Callback
# class ModelCheckpoint(tf.keras.callbacks.Callback):
#   def __init__(self, eval_dataset, metrics, save_model, saved_model_path):
#     super(ModelCheckpoint, self).__init__()
#     self.metrics = metrics
#     self.save_model = save_model
#     self.saved_model_path = saved_model_path
#     self.eval_dataset = eval_dataset
#     self.eval_labels: np.ndarray = None

#     for batch in eval_dataset:
#       if self.eval_labels is None:
#         self.eval_labels = batch[1].numpy()
#       else:
#         self.eval_labels = np.append(self.eval_labels, batch[1].numpy(), axis=0)

#   def evaluate(self):
#     eval_preds = []
#     for example in self.eval_dataset:
#       output = self.model(example[0])
#       eval_preds.append(tf.math.argmax(output.logits, axis=1))
    
#     eval_scores = {}
#     for metric_name, metric_func in self.metrics.items():
#       eval_scores[metric_name] = metric_func(self.eval_labels, eval_preds)
#     return eval_scores

#   def on_epoch_end(self, epoch, logs=None):
#     eval_scores = self.evaluate()
#     str_result = ""
#     for key, value in eval_scores.items():
#       str_result = str_result + " - {0}: {1:.4f}".format(key, value)
#     print(str_result)

#     if self.save_model:
#       self.model.save_weights(self.saved_model_path+"_"+str(epoch+1)+".h5")


# # Callback MNLI only
# class ModelCheckpointMNLI(tf.keras.callbacks.Callback):
#   def __init__(self, eval_dataset_matched, eval_dataset_mismatched, metrics, save_model, saved_model_path):
#     super(ModelCheckpointMNLI, self).__init__()
#     self.metrics = metrics
#     self.save_model = save_model
#     self.saved_model_path = saved_model_path
#     self.eval_dataset_matched = eval_dataset_matched
#     self.eval_dataset_mismatched = eval_dataset_mismatched
#     self.eval_labels_matched: np.ndarray = None
#     self.eval_labels_mismatched: np.ndarray = None

#     for batch in eval_dataset_matched:
#       if self.eval_labels_matched is None:
#         self.eval_labels_matched = batch[1].numpy()
#       else:
#         self.eval_labels_matched = np.append(self.eval_labels_matched, batch[1].numpy(), axis=0)

#     for batch in eval_dataset_mismatched:
#       if self.eval_labels_mismatched is None:
#         self.eval_labels_mismatched = batch[1].numpy()
#       else:
#         self.eval_labels_mismatched = np.append(self.eval_labels_mismatched, batch[1].numpy(), axis=0)

#   def evaluate(self):
#     eval_preds = {'matched': [], 'mismatched': []}
#     for example in self.eval_dataset_matched:
#       output = self.model(example[0])
#       eval_preds['matched'].append(tf.math.argmax(output.logits, axis=1))
#     for example in self.eval_dataset_mismatched:
#       output = self.model(example[0])
#       eval_preds['mismatched'].append(tf.math.argmax(output.logits, axis=1))
    
#     eval_scores = {'matched': {}, 'mismatched': {}}
#     for metric_name, metric_func in self.metrics.items():
#       eval_scores['matched'][metric_name] = metric_func(self.eval_labels_matched, eval_preds['matched'])
#       eval_scores['mismatched'][metric_name] = metric_func(self.eval_labels_mismatched, eval_preds['mismatched'])
#     return eval_scores

#   def on_epoch_end(self, epoch, logs=None):
#     eval_scores = self.evaluate()
#     str_result = "matched: "
#     for key, value in eval_scores['matched'].items():
#       str_result = str_result + " - {0}: {1:.4f}".format(key, value)
#     print(str_result)

#     str_result = "mismatched: "
#     for key, value in eval_scores['mismatched'].items():
#       str_result = str_result + " - {0}: {1:.4f}".format(key, value)
#     print(str_result)

#     if self.save_model:
#       self.model.save_weights(self.saved_model_path+"_"+str(epoch+1)+".h5")