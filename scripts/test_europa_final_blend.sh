#!/usr/bin/env bash
python ddp_test_nerf.py --config configs/europa/final.txt --render_splits static_path1_blend --test_env data/europa/final/static_path1_blend/envmaps
echo Finished
