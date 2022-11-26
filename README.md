# AdapLeR: Speeding up Inference by Adaptive Length Reduction

_Accepted as a conference paper for ACL 2022_

[Arxiv](https://arxiv.org/abs/2203.08991)

> **Abstract**: Pre-trained language models have shown stellar performance in various downstream tasks. But, this usually comes at the cost of high latency and computation, hindering their usage in resource-limited settings. In this work, we propose a novel approach for reducing the computational cost of BERT with minimal loss in downstream performance. Our method dynamically eliminates less contributing tokens through layers, resulting in shorter lengths and consequently lower computational cost. To determine the importance of each token representation, we train a Contribution Predictor for each layer using a gradient-based saliency method. Our experiments on several diverse classification tasks show speedups up to 22x during inference time without much sacrifice in performance. We also validate the quality of the selected tokens in our method using human annotations in the ERASER benchmark. In comparison to other widely used strategies for selecting important tokens, such as saliency and attention, our proposed method has a significantly lower false positive rate in generating rationales.

## Requirements

To install the required dependencies for this repo you can use `requirements.txt`:

```shell script
pip install -r requirements.txt
```

## Build Directories

Before fine-tuning and training AdapLeR, you need to create the directories where the fine-tuned model, saliency data, AdapLeR model, and logs are saved. For instance, for an sst2 task for seed 22:

```shell script
cd directory
./build_task_dirs.sh bert sst2 22
cd ..
```

## Fine-tune BERT

```shell script
python ./run_files/run_classification_bert.py --TASK sst2 --MAX_LENGTH 64 --DATA_SEED 22
```

## Extract Saliencies

```shell script
python ./tools/store_saliencies.py --TASK sst2 --DATA_SEED 22 --MAX_LENGTH 64 --BATCH_SIZE 4
```

## Training AdapLeR
```shell script
python ./run_files/run_classification_w_lr.py --TASK sst2 --GAMMA 0.005 --PHI 0.0005 --SAVE_NAME run_1_sst2 --BATCH_SIZE 32 --DATA_SEED 22 --EPOCHS 5 --MAX_LENGTH 64 --LEARNING_RATE 3e-5
```

## Evaluation
For those datasets which have a labelled test split:
```shell script
python ./run_files/run_classification_evaluation.py --TASK hatexplain --MAX_LENGTH 72 --BATCH_SIZE 48 --MODEL_PATH PATH_TO_MODEL.h5 --LR_MODEL
```
For GLUE tasks:
```shell script
python ./run_files/run_glue_prediction.py --TASK sst2 --MAX_LENGTH 64 --LR_MODEL --MODEL_PATH PATH_TO_MODEL.h5 --LR_MODEL
```

## Inference Mode
The evaluation method stated above employs a batchwise prediction loop and pre-computed FLOPs formulas for BERT and AdapLeR, which determines the final total speedup.

However to utilize AdapLeR in inference mode it is necessary to feed the model in a single instance manner (as stated in the paper). For this, the model can enter a length-reducing inference mode when the `lr_mode=True` flag is set:
```
encoded = tokenizer.encode_plus("a sample text.", return_tensors="tf")
outputs = model(encoded, lr_mode=True)
```
This mode will drop the non-contributing tokens in each layer, resulting in a lower computational cost than the vanilla BERT model.
