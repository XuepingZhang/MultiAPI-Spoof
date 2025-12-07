#!/bin/sh
source /dkucc/home/xz464/anaconda3/bin/activate
conda activate Antispoofing

python  ~/zxp/new_dataset_experiment_dkucc/aasist-main/main_Qwen_wishper.py\
    --config ~/zxp/new_dataset_experiment_dkucc/aasist-main/job/Qwen_whisper_aasist_with_new/eval_more/new/eval_new.conf \
    --outfile ~/zxp/new_dataset_experiment_dkucc/aasist-main/job/Qwen_whisper_aasist_with_new/eval_more/new/log/log.out\
    --eval
