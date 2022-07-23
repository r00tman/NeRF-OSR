#!/usr/bin/env bash
python ddp_test_nerf.py --config configs/trevi/final.txt --render_splits static_path1_blend --test_env data/trevi/final_clean/static_path1_blend/envmaps
echo Finished
