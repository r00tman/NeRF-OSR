#!/usr/bin/env bash
python ddp_test_nerf.py --config configs/europa/final.txt --render_splits static_path2 --test_env envs/15-08_14_30_IMG_8705.txt --rotate_test_env
echo Finished
