#!/usr/bin/env python3
import torch
import sys
import json
m = torch.load(sys.argv[1])

def format(x): return ',\n'.join('vec3({})'.format(', '.join(map(str, v.cpu().numpy()))) for v in x)

# print({k: format(v) for k, v in m['net_1'].items() if 'env_params' in k})
# for k,v in {k: v for k, v in m['net_1'].items() if 'env_params' in k}.items():
res = dict()
for k,v in {k: v for k, v in m['net_1'].items() if 'env_params' in k}.items():
    name = k[k.index('env_params.')+len('env_params.'):]
    res[name] = v.cpu().numpy().tolist()

    # print('---', k)
    # print(repr(v))
    # # print(format(v))
# print(res)
with open(sys.argv[1]+'.env_params.json', 'w') as f:
    json.dump(res, f, indent=2)

