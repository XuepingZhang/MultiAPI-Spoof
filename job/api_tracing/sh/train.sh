#!/bin/bash

(
while true; do
    nvidia-smi >> /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/api_tracing/log/nvidia-smi.out
    sleep 60
done
) &

python /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/main_api.py \
    --config /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/api_tracing/api_tracing.conf \
    --outfile /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/api_tracing/log/api_tracing_eval.out \
    --eval