import numpy as np
import tensorflow as tf
import itertools
import sys, os

PATH_TO_ERASER = '/home/username/NLP/ERASER'
sys.path.append(os.path.abspath(PATH_TO_ERASER))
from eraserbenchmark.rationale_benchmark.utils import load_documents, load_datasets

eraser_tasks = ['movie_rationales', 'boolq']

class_mapper = {
    'NEG': 0,
    'POS': 1,
    'False': 0,
    'True': 1,
}
num_label_tasks = {
  'movie_rationales': 2,
}

def eraser_to_token_level_rationale_data(data_root):
  documents = load_documents(data_root)
  raw_train, raw_eval, raw_test = load_datasets(data_root)

  data = {'train': [], 'validation': [], 'test': []}
  for split, raw_data in zip(data.keys(), [raw_train, raw_eval, raw_test]):
    for ex in range(len(raw_data)):
      ann = raw_data[ex]
      evidences = ann.all_evidences()
      if len(evidences) == 0:
        continue

      (docid,) = set(ev.docid for ev in evidences)
      doc = documents[docid]
      flattened_doc = list(itertools.chain.from_iterable(doc))
      rationale = np.zeros(len(flattened_doc), dtype=np.int8)
      
      for e, ev in enumerate(evidences):
        rationale[ev.start_token:ev.end_token] = e + 1

      data[split].append({'query': ann.query.split(), 'text': flattened_doc, 'rationale':rationale, 'label':class_mapper[ann.classification]})
  
  return data


def movies_to_features(data,
                       tokenizer,
                       max_length,
                       return_all=False,
                      ):

  def gen():
    length = max_length - 2
    for ex, example in enumerate(data):
      total_tokens = []
      total_word_index = []
      for index, word in enumerate(example['text']):
        tokenized_out = tokenizer.encode(word, add_special_tokens=False)
        total_tokens.extend(tokenized_out)
        total_word_index.extend([index] * len(tokenized_out))
      total_word_index = np.array(total_word_index)

      total_rationale = np.zeros(len(total_tokens))
      for ev_id in range(np.max(example['rationale'])):
        condition = np.where(example['rationale'] == ev_id+1)[0]
        for c in condition:
          total_rationale[np.where(total_word_index == c)[0]] = ev_id+1

      tail = max_length - 2
      exit_flag = False
      while not exit_flag:
        head = tail - length
        word_index = total_word_index[head:tail]
        if tail >= len(total_word_index):
          tail = len(total_word_index)
          exit_flag = True
        for i in reversed(range(tail)): # kepp fully a rationale span
          if i + 1 > tail - 1:
            break #last segment of text
          if total_rationale[i+1] == 0 or total_rationale[i] != total_rationale[i+1]:
            break
          tail -= 1
        
        token_ids = total_tokens[head:tail]
        word_index = total_word_index[head:tail]
        rationale = total_rationale[head:tail]
        
        start_token_index, end_token_index = head, tail
        # zero-start word indexing
        if head != 0:
          word_index = word_index - total_word_index[head]
        
        tail += length

        if np.all((rationale == 0)):
          continue

        # padding
        input_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (max_length - 2 - len(token_ids))
        attention_mask = [1] * (len(token_ids) + 2) + [0] * (max_length - 2 - len(token_ids))
        rationale = np.concatenate(([-1], rationale, [-1], [-1] * (max_length - 2 - len(token_ids))))
        word_index = np.concatenate(([-1], word_index, [-1], [-1] * (max_length - 2 - len(token_ids))))

        if return_all:
          yield ({'input_ids': input_ids,
            'attention_mask': attention_mask},
            {'label': example['label'], 
              'rationale': rationale, 
              'word_index': word_index,
              'start_token_index': start_token_index,
              'end_token_index': end_token_index,
              'example_id': ex})
        else:
          yield ({'input_ids': input_ids,
                'attention_mask': attention_mask},
                example['label'])


  return tf.data.Dataset.from_generator(gen,
      ({'input_ids': tf.int32,
        'attention_mask': tf.int32},
       {'label': tf.int32, 
        'rationale': tf.int32, 
        'word_index': tf.int32,
        'start_token_index':tf.int32,
        'end_token_index':tf.int32,
        'example_id':tf.int32}) if return_all else 
        ({'input_ids': tf.int32,
        'attention_mask': tf.int32},
        tf.int32),
      ({'input_ids': tf.TensorShape([None]),
        'attention_mask': tf.TensorShape([None])},
       {'label': tf.TensorShape([]), 
        'rationale': tf.TensorShape([None]), 
        'word_index': tf.TensorShape([None]),
        'start_token_index':tf.TensorShape([]),
        'end_token_index': tf.TensorShape([]),
        'example_id': tf.TensorShape([])}) if return_all else 
        ({'input_ids': tf.TensorShape([None]),
        'attention_mask': tf.TensorShape([None])},
        tf.TensorShape([]))
        )

# @title Convert to Rationale Features
def only_rationales_to_features(data,
                     tokenizer,
                     max_length,
                     ):
  
  def gen():
    for example in data:
      token_ids = []
      word_indices = []
      for index, word in enumerate(example['text']):
        tokenized_out = tokenizer.encode(word, add_special_tokens=False)
        token_ids.extend(tokenized_out)
        word_indices.extend([index] * len(tokenized_out))
      word_indices = np.array(word_indices)

      rationale = np.zeros(len(token_ids))
      condition = np.where(example['rationale'] == 1)[0]
      for c in condition:
        rationale[np.where(word_indices == c)[0]] = 1

      if np.sum(rationale) > max_length - 2:
        continue

      position_ids = np.where(rationale == 1)[0]
      token_ids = np.array(token_ids)[position_ids]
      word_indices = word_indices[position_ids]

      # padding
      input_ids = np.concatenate(([tokenizer.cls_token_id], token_ids, [tokenizer.sep_token_id], [tokenizer.pad_token_id] * (max_length - 2 - len(token_ids))))
      attention_mask = [1] * (len(token_ids) + 2) + [0] * (max_length - 2 - len(token_ids))
      position_ids = np.concatenate(([0], position_ids, range(position_ids[-1]+1, max_length - len(position_ids) + position_ids[-1]+1-1)))
      # rationale = np.concatenate(([-1], rationale, [-1], [-1] * (max_length - 2 - len(token_ids))))
      # word_index = np.concatenate(([-1], word_indices, [-1], [-1] * (max_length - 2 - len(token_ids))))


      yield ({'input_ids': input_ids,
              'attention_mask': attention_mask,
              'position_ids': position_ids},
              example['label'])

  return tf.data.Dataset.from_generator(gen,
      ({'input_ids': tf.int32,
        'attention_mask': tf.int32,
        'position_ids': tf.int32},
       tf.int32),
      ({'input_ids': tf.TensorShape([None]),
        'attention_mask': tf.TensorShape([None]),
        'position_ids': tf.TensorShape([None])},
       tf.TensorShape([])))


convertors = {
    "movie_rationales": movies_to_features,
}
