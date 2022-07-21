#!/usr/bin/env python3
import torch
import sys
m = torch.load(sys.argv[1])

def format(x): return ',\n'.join('vec3({})'.format(', '.join(map(str, v.cpu().numpy()))) for v in x)

# print({k: format(v) for k, v in m['net_1'].items() if 'env_params' in k})
# for k,v in {k: v for k, v in m['net_1'].items() if 'env_params' in k}.items():
for k,v in {k: v for k, v in m['net_1'].items() if 'env' in k}.items():
    print('---', k)
    print(repr(v))
    # print(format(v))

