	reset
    python3 policy_compression_train.py \
    --log_dir "evals/log_dagger" \
    --policy_dir "evals/tmp_dagger" \
    --eval-ep-len 4 \
    --dagger-beta 35 \
    --n_iter 5 \
    --n_eval 5 \
    --seed 1

