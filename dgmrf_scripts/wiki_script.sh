#!/bin/sh

mkdir wiki_dgmrf_logs

for base_ds in chameleon squirrel crocodile
do
    ds=wiki_$base_ds\_random_0.5
    for seed in 913 914 915 916 917
    do
        # 1, 3 layers 80K iterations
        for L in 1 3
        do
            log_file=./wiki_dgmrf_logs/$base_ds\_$L\_$seed.txt
            python -u main.py --dataset $ds --n_layers $L --seed $seed --vi_layers 1 --n_iterations 80000 | tee $log_file
        done

        # 5 layers 150K iterations
        L=5
        log_file=./wiki_dgmrf_logs/$base_ds\_$L\_$seed.txt
        python -u main.py --dataset $ds --n_layers $L --seed $seed --vi_layers 1 --n_iterations 150000 | tee $log_file
    done
done
