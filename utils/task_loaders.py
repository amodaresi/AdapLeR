import sys
import os

sys.path.append("..")

import json
import numpy as np
from random import Random

import tensorflow as tf
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix

from transformers import BertTokenizer

from utils.classification_utils import text_examples_to_tfdataset
from utils.glue_utils import glue_examples_to_tfdataset, glue_tasks_metrics, glue_tasks_num_labels

from directories import *

def sals_to_tfdataset(sals):
    def gen():
        for i in range(len(sals)):
            yield sals[i]

    return tf.data.Dataset.from_generator(gen,
        tf.float32,
        tf.TensorShape([12, None]))

def get_best_epochs(dir, return_scores=False):
    ### Returns epochs from worst to best
    with open(dir, "r", encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    dev_set = list(data[0].keys())[0]
    dev_metric = list(data[0][dev_set].keys())[0]
    scores = []
    for i in range(len(data)):
        scores.append(data[i][dev_set][dev_metric])
    if return_scores:
        return scores
    return np.argsort(scores)


def load_imdb(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    train_data = load_dataset("imdb", split='train').shuffle(seed=seed)
    test_data = load_dataset("imdb", split='test')

    num_train_examples = len(train_data) - 4096
    num_eval_examples = 4096
    num_test_examples = len(test_data)

    train_dataset = text_examples_to_tfdataset(train_data, tokenizer, max_length=max_length).cache()
    eval_dataset = train_dataset.take(4096)
    train_dataset = train_dataset.skip(4096)
    test_dataset = text_examples_to_tfdataset(test_data, tokenizer, max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": train_data.features["label"].num_classes
        },
        {
            'accuracy': accuracy_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("imdb", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/imdb/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_agnews(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    train_data = load_dataset("ag_news", split='train').shuffle(seed=seed)
    test_data = load_dataset("ag_news", split='test')

    num_train_examples = len(train_data) - 6000
    num_eval_examples = 6000
    num_test_examples = len(test_data)

    train_dataset = text_examples_to_tfdataset(train_data, tokenizer, max_length=max_length).cache()
    eval_dataset = train_dataset.take(6000)
    train_dataset = train_dataset.skip(6000)
    test_dataset = text_examples_to_tfdataset(test_data, tokenizer, max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": train_data.features["label"].num_classes
        },
        {
            'accuracy': accuracy_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("ag_news", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/ag_news/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)
        # sal_dataset = sals_to_tfdataset(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_dbpedia(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    train_data = load_dataset("dbpedia_14", split='train').shuffle(seed=seed)
    test_data = load_dataset("dbpedia_14", split='test')

    num_train_examples = len(train_data) - 6000
    num_eval_examples = 6000
    num_test_examples = len(test_data)

    train_dataset = text_examples_to_tfdataset(train_data, tokenizer, max_length=max_length, text_keyname="content").cache()
    eval_dataset = train_dataset.take(6000)
    train_dataset = train_dataset.skip(6000)
    test_dataset = text_examples_to_tfdataset(test_data, tokenizer, max_length=max_length, text_keyname="content").cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": train_data.features["label"].num_classes
        },
        {
            'accuracy': accuracy_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("dbpedia", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/dbpedia/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)
        # sal_dataset = sals_to_tfdataset(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_scitail(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    def update_label(example):
        example["label"] = 0 if example["label"] == "neutral" else 1
        return example

    train_data = load_dataset("scitail", "tsv_format", split='train').map(update_label).shuffle(seed=seed)
    eval_data = load_dataset("scitail", "tsv_format", split='validation').map(update_label)
    test_data = load_dataset("scitail", "tsv_format", split='test').map(update_label)

    num_train_examples = len(train_data)
    num_eval_examples = len(eval_data)
    num_test_examples = len(test_data)

    train_dataset = glue_examples_to_tfdataset(train_data, tokenizer, "mnli", max_length=max_length).cache()
    eval_dataset = glue_examples_to_tfdataset(eval_data, tokenizer, "mnli", max_length=max_length).cache()
    test_dataset = glue_examples_to_tfdataset(test_data, tokenizer, "mnli", max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": 2
        },
        {
            'accuracy': accuracy_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("scitail", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/scitail/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_amazon(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    def update_label(example):
        example["label"] = example["stars"] - 1
        return example

    train_data = load_dataset("amazon_reviews_multi", "en", split='train').map(update_label).shuffle(seed=seed)
    eval_data = load_dataset("amazon_reviews_multi", "en", split='validation').map(update_label)
    test_data = load_dataset("amazon_reviews_multi", "en", split='test').map(update_label)

    num_train_examples = len(train_data)
    num_eval_examples = len(eval_data)
    num_test_examples = len(test_data)

    train_dataset = text_examples_to_tfdataset(train_data, tokenizer, max_length=max_length, text_keyname="review_body").cache()
    eval_dataset = text_examples_to_tfdataset(eval_data, tokenizer, max_length=max_length, text_keyname="review_body").cache()
    test_dataset = text_examples_to_tfdataset(test_data, tokenizer, max_length=max_length, text_keyname="review_body").cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": 5
        },
        {
            'accuracy': accuracy_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("amazon", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/amazon/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)
        # sal_dataset = sals_to_tfdataset(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_hatexplain(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    def mode(lst):
        return max(set(lst), key=lst.count)
    
    def update_data(example):
        example["label"] = mode(example["annotators"]["label"])
        example["text"] = " ".join(example["post_tokens"])
        return example

    train_data = load_dataset("hatexplain", split='train').map(update_data).shuffle(seed=seed)
    eval_data = load_dataset("hatexplain", split='validation').map(update_data)
    test_data = load_dataset("hatexplain", split='test').map(update_data)

    num_train_examples = len(train_data)
    num_eval_examples = len(eval_data)
    num_test_examples = len(test_data)

    train_dataset = text_examples_to_tfdataset(train_data, tokenizer, max_length=max_length).cache()
    eval_dataset = text_examples_to_tfdataset(eval_data, tokenizer, max_length=max_length).cache()
    test_dataset = text_examples_to_tfdataset(test_data, tokenizer, max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": 3
        },
        {
            'accuracy': accuracy_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("hatexplain", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/hatexplain/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)
        # sal_dataset = sals_to_tfdataset(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_eraser_task(task, tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    from utils.rationale_utils import eraser_to_token_level_rationale_data, num_label_tasks
    from utils.rationale_utils import convertors as rationale_convertors
    
    data_root = os.path.join('../ERASER/eraserbenchmark/data', task)
    data = eraser_to_token_level_rationale_data(data_root)
    train_data = data['train']
    eval_data = data['validation']
    test_data = data['test']

    Random(seed).shuffle(train_data)

    train_dataset = rationale_convertors[task](train_data, tokenizer, max_length=max_length).cache()
    eval_dataset = rationale_convertors[task](eval_data, tokenizer, max_length=max_length).cache()
    test_dataset = rationale_convertors[task](test_data, tokenizer, max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size, drop_remainder=True)
    eval_dataset = eval_dataset.batch(eval_batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(eval_batch_size, drop_remainder=True)

    train_steps = int(3178 // training_batch_size) if task == "movie_rationales" else len(list(train_dataset.as_numpy_iterator()))
    print(train_steps)
    eval_steps = int(388 // eval_batch_size) if task == "movie_rationales" else len(list(eval_dataset.as_numpy_iterator()))
    print(eval_steps)
    test_steps = int(423 // eval_batch_size) if task == "movie_rationales" else len(list(test_dataset.as_numpy_iterator()))
    print(test_steps)
    num_train_examples = train_steps * training_batch_size

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "val_steps": eval_steps,
            "val_examples": eval_steps * eval_batch_size,
            "test_steps": test_steps,
            "test_examples": test_steps * eval_batch_size,
            "num_labels": 2
        },
        {
            'accuracy': accuracy_score,
            "f1": f1_score
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ(task, "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/{task}/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)
        # sal_dataset = sals_to_tfdataset(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_glue_task(task, tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    train_data = load_dataset("glue", task , split='train').shuffle(seed=seed)
    if task != "mnli":
        validation_data = load_dataset("glue", task, split='validation')
        test_data = load_dataset("glue", task, split='test')
    else:
        validation_m_data = load_dataset("glue", task, split='validation_matched')
        validation_mm_data = load_dataset("glue", task, split='validation_mismatched')
        test_m_data = load_dataset("glue", task, split='test_matched')
        test_mm_data = load_dataset("glue", task, split='test_mismatched')
    
    num_train_examples = len(train_data)

    train_dataset = glue_examples_to_tfdataset(train_data, tokenizer, task, max_length=max_length).cache()

    if task != "mnli":
        validation_dataset = glue_examples_to_tfdataset(validation_data, tokenizer, task, max_length=max_length).cache()
        test_dataset = glue_examples_to_tfdataset(test_data, tokenizer, task, max_length=max_length).cache()
    else:
        validation_m_dataset = glue_examples_to_tfdataset(validation_m_data, tokenizer, task, max_length=max_length).cache()
        validation_mm_dataset = glue_examples_to_tfdataset(validation_mm_data, tokenizer, task, max_length=max_length).cache()
        test_m_dataset = glue_examples_to_tfdataset(test_m_data, tokenizer, task, max_length=max_length).cache()
        test_mm_dataset = glue_examples_to_tfdataset(test_mm_data, tokenizer, task, max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size)

    if task != "mnli":
        validation_dataset = validation_dataset.batch(eval_batch_size)
        test_dataset = test_dataset.batch(eval_batch_size)
    else:
        validation_m_dataset = validation_m_dataset.batch(eval_batch_size)
        validation_mm_dataset = validation_mm_dataset.batch(eval_batch_size)
        test_m_dataset = test_m_dataset.batch(eval_batch_size)
        test_mm_dataset = test_mm_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    
    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": validation_dataset} if task != "mnli" else {"validation-m": validation_m_dataset, "validation-mm": validation_mm_dataset},
            "tests": {"test": test_dataset} if task != "mnli" else {"test-m": test_m_dataset, "test-mm": test_mm_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": glue_tasks_num_labels[task]
        },
        glue_tasks_metrics[task]
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ(task, "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/{task}/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_fnc(tokenizer, max_length, training_batch_size, eval_batch_size, seed=42, include_sals=False):
    base_url = "https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/"
    stances_dataset = load_dataset("csv", data_files={'train': base_url + 'train_stances.csv', 'test': base_url + 'competition_test_stances.csv'})
    bodies_dataset = load_dataset("csv", data_files={'train': base_url + 'train_bodies.csv', 'test': base_url + 'competition_test_bodies.csv'})

    body_id_to_idx = {"train": {}, "test": {}}
    for k in body_id_to_idx.keys():
        for i, v in enumerate(bodies_dataset[k]):
            body_id_to_idx[k][v["Body ID"]] = i

    label_to_id = {
        "unrelated": 0,
        "discuss": 1,
        "agree": 2,
        "disagree": 3
    }

    def update_data_train(example):
        example["label"] = label_to_id[example["Stance"]]
        example["premise"] = example["Headline"]
        example["hypothesis"] = bodies_dataset["train"][body_id_to_idx["train"][example["Body ID"]]]["articleBody"]
        return example

    def update_data_test(example):
        example["label"] = label_to_id[example["Stance"]]
        example["premise"] = example["Headline"]
        example["hypothese"] = bodies_dataset["test"][body_id_to_idx["test"][example["Body ID"]]]["articleBody"]
        return example

    train_data = stances_dataset["train"].map(update_data_train).shuffle(seed=seed)
    test_data = stances_dataset["test"].map(update_data_test)

    num_train_examples = len(train_data) - 4096
    num_eval_examples = 4096
    num_test_examples = len(test_data)

    train_dataset = glue_examples_to_tfdataset(train_data, tokenizer, "mnli", max_length=max_length).cache()
    eval_dataset = train_dataset.take(4096)
    train_dataset = train_dataset.skip(4096)
    test_dataset = glue_examples_to_tfdataset(train_data, tokenizer, "mnli", max_length=max_length).cache()

    train_dataset = train_dataset.batch(training_batch_size)
    eval_dataset = eval_dataset.batch(eval_batch_size)
    test_dataset = test_dataset.batch(eval_batch_size)

    train_steps = int(np.ceil(num_train_examples / training_batch_size))
    eval_steps = int(np.ceil(num_eval_examples / eval_batch_size))
    test_steps = int(np.ceil(num_test_examples / eval_batch_size))

    def fnc_score_mean(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_t_rel = 1 * (y_true != 0)
        y_p_rel = 1 * (y_pred != 0)
        score_1 = np.sum(y_t_rel == y_p_rel)
        score_2 = np.sum(y_true[y_true != 0] == y_pred[y_true != 0])

        max_score = 0.25 * len(y_t_rel) + 0.75 * len(y_true[y_true != 0])

        return (0.25 * (score_1) + 0.75 * score_2) / max_score

    def fnc_score(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_t_rel = 1 * (y_true != 0)
        y_p_rel = 1 * (y_pred != 0)
        score_1 = np.sum(y_t_rel == y_p_rel)
        score_2 = np.sum(y_true[y_true != 0] == y_pred[y_true != 0])

        return 0.25 * (score_1) + 0.75 * score_2

    def fnc_reltype_acc(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        score_2 = accuracy_score(y_true[y_true != 0], y_pred[y_true != 0])
        return score_2

    def fnc_rel_acc(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_t_rel = 1 * (y_true != 0)
        y_p_rel = 1 * (y_pred != 0)
        score_1 = accuracy_score(y_t_rel, y_p_rel)
        print("baseline_rel_acc:" , accuracy_score(y_t_rel, np.zeros_like(y_p_rel)))
        print("CM:", confusion_matrix(y_true, y_pred))

        return score_1

    def score_submission(gold_labels, test_labels):
        score = 0.0
        cm = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]

        for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
            if g == t:
                score += 0.25
                if g != 0:
                    score += 0.50
            if g != 0 and t != 0:
                score += 0.25
            
            cm[g][t] += 1

        print("CM: ", cm)
        return score

    outputs = (
        {
            "train": train_dataset,
            "evals": {"validation": eval_dataset},
            "tests": {"test": test_dataset}
        },
        {
            "train_steps": train_steps,
            "train_examples": num_train_examples,
            "num_labels": 4,
            "class_weights": np.array([1., 3., 3., 3.])
        },
        {
            'fnc_relative_score': fnc_score_mean,
            'fnc_score': fnc_score,
            'fnc_reltype_acc': fnc_reltype_acc,
            'fnc_rel_acc': fnc_rel_acc,
            'official_scoring_system': score_submission
        }
    )

    if include_sals:
        train_summed_sal_mat = np.zeros((num_train_examples, max_length), dtype=np.float32)
        SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_READ("fnc", "bert" if tokenizer.name_or_path[:4] == "bert" else "albert", seed=seed)
        BEST_EPOCHS = get_best_epochs(f"./directory/bert/{seed}/fnc/logs/ft_logs.json")[-3:] + 1

        for i in BEST_EPOCHS:
            loaded = np.load(SALIENCIES_PATH+f"{i}.npy")
            if loaded.ndim == 3:
                train_summed_sal_mat += loaded[:num_train_examples, 0, :max_length]
            else:
                train_summed_sal_mat += loaded[:num_train_examples, :max_length]
            
        train_summed_sal_mat /= len(BEST_EPOCHS)

        sal_dataset = tf.data.Dataset.from_tensor_slices(train_summed_sal_mat).batch(training_batch_size)
        # sal_dataset = sals_to_tfdataset(train_summed_sal_mat).batch(training_batch_size)

        outputs = outputs + (sal_dataset,)

    return outputs

def load_movie(**kwargs):
    return load_eraser_task("movie_rationales", **kwargs)

def load_sst2(**kwargs):
    return load_glue_task("sst2", **kwargs)

def load_qnli(**kwargs):
    return load_glue_task("qnli", **kwargs)

load_task = {
    "imdb" : load_imdb,
    "ag_news": load_agnews,
    "movie_rationales": load_movie,
    "dbpedia": load_dbpedia,
    "amazon": load_amazon,
    "scitail": load_scitail,
    "hatexplain": load_hatexplain,
    "fnc": load_fnc,
    "sst2": load_sst2,
    "qnli": load_qnli
}
