#!/bin/sh

mkdir fraction_baseline_logs

for model in igmrf graphgp
do
    for frac in 0.05 0.2 0.4 0.6 0.8 0.95
    do
        # Mix
        ds=gmrf_prec_mix32_random_$frac
        log_file=./fraction_baseline_logs/$model\_$ds.txt
        python -u eval_baseline.py --model $model --dataset $ds --pos 0 --features 0 | tee $log_file

        # Crocodile
        ds=wiki_crocodile_random_$frac
        log_file=./fraction_baseline_logs/$model\_$ds.txt
        python -u eval_baseline.py --model $model --dataset $ds --pos 0 --features 0 --igmrf_eps 1e-4 | tee $log_file
    done
done
