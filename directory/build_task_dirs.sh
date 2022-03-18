#!/bin/bash
if [[ $1 = "bert" ]]; then
    echo "Building directories for MODEL: $1 and TASK: $2 and SEED: $3"
    if [ ! -d "bert" ]; then
        mkdir bert
    fi
    cd bert
    mkdir $3
    cd $3
    mkdir $2
    cd $2
else
    echo "BAD INPUT"
    exit 2
fi

mkdir models
mkdir models/adaptive_lr
mkdir logs
mkdir data

mkdir data/sals
mkdir data/attn

mkdir data/sals/train
mkdir data/sals/test
mkdir data/sals/val

mkdir data/attn/train
mkdir data/attn/test
mkdir data/attn/val
