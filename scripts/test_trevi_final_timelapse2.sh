#!/usr/bin/env bash
python ddp_test_nerf.py --config configs/trevi/final.txt --render_splits static_path2 --test_env envs/97557154_3618766093.txt --rotate_test_env
echo Finished
