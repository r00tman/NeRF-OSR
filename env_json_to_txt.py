#!/usr/bin/env python3
import numpy as np
import json
import sys

which = sys.argv[2]
dest = sys.argv[3]

with open(sys.argv[1], 'r') as f:
    envs = json.load(f)

envs = {x: y for x, y in envs.items() if which.lower() in x.lower()}
print('found', envs)
assert(len(envs) == 1)

env = list(envs.values())[0]
env = np.array(env)

np.savetxt(dest, env)
