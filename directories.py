def WORKER_MODEL_PATH(TASK, backbone, seed=1):
    return f"./directory/{backbone}/{seed}/{TASK}/models/{backbone}_"+TASK+"_"

def WORKER_MODEL_LOAD_PATH(TASK, backbone, NAME, seed=1):
    return f"./directory/{backbone}/{seed}/{TASK}/models/{NAME}.h5"


def WORKER_MODEL_SAVE_PATH(TASK, backbone, SAVE_NAME, seed=1, **kwargs):
    sub_name = ""
    for key, value in kwargs.items():
        sub_name = sub_name + "_" + key + str(value)
    return f"./directory/{backbone}/{seed}/{TASK}/models/adaptive_lr/{SAVE_NAME}" + sub_name

def WORKER_LOG_SAVE_PATH(TASK, backbone, SAVE_NAME, seed=1, **kwargs):
    sub_name = ""
    for key, value in kwargs.items():
        sub_name = sub_name + "_" + key + str(value)
    return f"./directory/{backbone}/{seed}/{TASK}/logs/{SAVE_NAME}" + sub_name


def WORKER_SALIENCIES_PATH_to_WRITE(TASK, backbone, split="train", seed=1):
    return f"./directory/{backbone}/{seed}/{TASK}/data/sals/{split}/"

def WORKER_SALIENCIES_PATH_to_READ(TASK, backbone, split="train", seed=1):
    return f"./directory/{backbone}/{seed}/{TASK}/data/sals/{split}/sal_mat_wCLSSEPnoZ_"

def WORKER_ATTENTIONS_PATH_to_READ(TASK, backbone, split="train", name='attn_wCLSSEP_', seed=1):
    return f"./directory/{backbone}/{seed}/{TASK}/data/attn/{split}/{name}"

def WORKER_ATTENTIONS_PATH_to_WRITE(TASK, backbone, split="train", seed=1):
    return f"./directory/{backbone}/{seed}/{TASK}/data/attn/{split}/"

LONG_BERT_DIR = "./directory/bert/bert-base-uncased-1024"