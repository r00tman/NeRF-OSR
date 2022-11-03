#!/usr/bin/env python3
import numpy as np
import os
import sys
import json
import shutil

# cameras = json.load(open('./kai_cameras_offsetnorm.json'))
# cameras = json.load(open('./kai_cameras.json'))
cameras = json.load(open('./kai_cameras_normalized.json'))

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def w2c_to_c2w(w2c):
    w2c = np.reshape(w2c, (4, 4))
    c2w = np.linalg.inv(w2c)
    #c2w = convert_pose(c2w)
    return c2w.reshape(-1)

for base in ['train', 'test', 'validation']:
    obj = dict()
    os.makedirs(os.path.join(base, 'intrinsics'), exist_ok=True)
    os.makedirs(os.path.join(base, 'pose'), exist_ok=True)
    os.makedirs(os.path.join(base, 'mask'), exist_ok=True)
    for c in cameras:
        fn = '.'.join(c.split('.')[:-1])+'.txt'
        if os.path.exists(os.path.join(base, 'rgb', c)):
            print(base, fn)
            with open(os.path.join(base, 'intrinsics', fn), 'w') as f:
                print(*cameras[c]['K'], file=f)
            with open(os.path.join(base, 'pose', fn), 'w') as f:
                w2c = cameras[c]['W2C']
                c2w = w2c_to_c2w(w2c)
                print(*c2w, file=f)
            maskfn = os.path.join('mask', '.'.join(c.split('.')[:-1])+'.png')
            shutil.copyfile(maskfn, os.path.join(base, maskfn))

            obj[c] = cameras[c]
    with open(os.path.join(base, 'cam_dict_norm.json'), 'w') as f:
        json.dump(obj, f)
