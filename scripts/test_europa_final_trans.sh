#!/usr/bin/env bash
python ddp_test_nerf.py --config configs/europa/final.txt --render_splits trans_path --test_env envs/15-08_14_30_IMG_8705.txt
echo Finished
