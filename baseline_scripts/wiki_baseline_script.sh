#!/bin/sh

mkdir wiki_baseline_logs

for model in lp igmrf dgp graphgp
do
    for base_ds in chameleon squirrel crocodile
    do
        ds=wiki_$base_ds\_random_0.5
        log_file=./wiki_baseline_logs/$model\_$base_ds.txt
        python -u eval_baseline.py --model $model --dataset $ds --pos 0 --features 0 --igmrf_eps 1e-4 | tee $log_file
    done
done
