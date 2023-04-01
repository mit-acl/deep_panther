	reset
    python3 policy_compression_train.py \
    --net_arch 512 512
    # --log_dir "evals/log_dagger" \
    # --policy_dir "evals/tmp_dagger" \
    # --eval_ep_len 200 \
    # --dagger_beta 200 \
    # --n_iters 10 \
    # --n_eval 6 \
    # --seed 1