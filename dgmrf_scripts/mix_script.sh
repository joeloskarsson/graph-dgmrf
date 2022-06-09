#!/bin/sh

mkdir mix_dgmrf_logs

ds=gmrf_prec_mix32_random
for seed in 913 914 915 916 917
do
    for L in 1 2 3 4 5
    do
        log_file=./mix_dgmrf_logs/dgmrf_$L\_$seed.txt
        python -u main.py --dataset $ds --n_layers $L --seed $seed --vi_layers 1 --n_iterations 50000 | tee $log_file
    done
done
