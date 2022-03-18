import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

MODEL_PATH = 'bert-base-uncased'
num_labels = 2

from transformers import (
    BertConfig,
)

from modeling_tf_bert import TFBertForSequenceClassification

# Load Model
config = BertConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
model.to_ALR()


def get_flops(input_structure):
    concrete = tf.function(lambda inputs: model(inputs, lr_mode=True))
    concrete_func = concrete.get_concrete_function(
        input_structure)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


max_length = 8
flops = np.empty((13, max_length-1))
for length in tqdm(range(2, max_length+1)):
    input_structure = {
        "input_ids": tf.TensorSpec([1, length], dtype=tf.int32), 
        "attention_mask": tf.TensorSpec([1, length], dtype=tf.int32), 
        "token_type_ids": tf.TensorSpec([1, length], dtype=tf.int32)
        }
    for l in range(13):
        print(l)
        model.bert.encoder.num_hidden_layers = l
        model.bert.config.num_hidden_layers = l
        model.config.num_hidden_layers = l

        flops[l, length - 2] = get_flops(input_structure)

polys = []
for l in range(13):
    polys.append(np.polyfit(np.arange(2, max_length+1), flops[l], 2))

print(polys)
print("Bias: ", polys[-1][2])
print("O(n) bias: ", polys[0][1])
print("Each layer O(n^2) coeff: ", polys[-1][0] - polys[-2][0])
print("Each layer O(n) coeff: ", polys[-1][1] - polys[-2][1])