#!/bin/sh

mkdir dgp_fraction_seed_logs

# Croc
for frac in 0.05 0.2 0.4 0.6 0.8 0.95
do
    for seed in 1000 2000 3000 4000 5000
    do
        ds=wiki_crocodile_random_$frac
        log_file=./dgp_fraction_seed_logs/croc_$frac\_$seed.txt
        python -u eval_baseline.py --model dgp --dataset $ds --pos 0 --features 0 --seed $seed --gnn_config 5 | tee $log_file
    done
done

# Mix
for frac in 0.05 0.2 0.4 0.6 0.8 0.95
do
    for seed in 1000 2000 3000 4000 5000
    do
        ds=gmrf_prec_mix32_random_$frac
        log_file=./dgp_fraction_seed_logs/mix_$frac\_$seed.txt
        python -u eval_baseline.py --model dgp --dataset $ds --pos 0 --features 0 --seed $seed --gnn_config 8 | tee $log_file
    done
done

