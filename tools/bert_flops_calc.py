import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

MODEL_PATH = 'bert-base-uncased'
num_labels = 2

from transformers import (
    BertConfig,
    TFBertForSequenceClassification
)

# Load Model
config = BertConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)

def get_flops(input_structure):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        input_structure)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


max_length = 5
flops = np.empty(max_length-1)
for length in tqdm(range(2, max_length+1)):
    input_structure = {
        "input_ids": tf.TensorSpec([1, length], dtype=tf.int32), 
        "attention_mask": tf.TensorSpec([1, length], dtype=tf.int32), 
        "token_type_ids": tf.TensorSpec([1, length], dtype=tf.int32)
        }
    flops[length - 2] = get_flops(input_structure)

polys = []
polys.append(np.polyfit(np.arange(2, max_length+1), flops, 2))

print(polys)