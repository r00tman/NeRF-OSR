import numpy as np
import torch
# from torch import nn, optim
# import torch.nn.functional as F
from demo_projSH_rotSH import Rotation

def illuminate_vec_old(n, env):
    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743125
    c4 = 0.886227
    c5 = 0.247708
    c = env.unsqueeze(1)
    x, y, z = n[..., 0, None], n[..., 1, None], n[..., 2, None]
    irradiance = (
        c1 * c[8] * (x ** 2 - y ** 2) +
        c3 * c[6] * (z ** 2) +
        c4 * c[0] -
        c5 * c[6] +
        2 * c1 * c[4] * x * y +
        2 * c1 * c[7] * x * z +
        2 * c1 * c[5] * y * z +
        2 * c2 * c[3] * x +
        2 * c2 * c[1] * y +
        2 * c2 * c[2] * z
    )
    return irradiance

def illuminate_vec(n, env):
    c1 = 0.282095
    c2 = 0.488603
    c3 = 1.092548
    c4 = 0.315392
    c5 = 0.546274

    c = env.unsqueeze(1)
    x, y, z = n[..., 0, None], n[..., 1, None], n[..., 2, None]

    irradiance = (
        c[0] * c1 +
        c[1] * c2*y +
        c[2] * c2*z +
        c[3] * c2*x +
        c[4] * c3*x*y +
        c[5] * c3*y*z +
        c[6] * c4*(3*z*z-1) +
        c[7] * c3*x*z +
        c[8] * c5*(x*x-y*y)
    )
    return irradiance

def rotate_vec(v, a):
    c = np.cos(a)
    s = np.sin(a)

    x = v[..., 0]*c-v[..., 2]*s
    y = v[..., 1]
    z = v[..., 0]*s+v[..., 2]*c

    res = torch.stack((x, y, z), -1)
    return res

def rotate_env(env, angle):
    rotation = Rotation()
    rot = np.float32(np.dot(rotation.rot_y(angle), np.dot(rotation.rot_x(0.), rotation.rot_z(0.))))
    rot_sh = np.matmul(rot, env.clone().detach().cpu().numpy())

    return torch.tensor(rot_sh).to(env.device)

# def rotate_env(env, angle):
#     c1 = 0.429043
#     c2 = 0.511664
#     c3 = 0.743125
#     c4 = 0.886227
#     c5 = 0.247708
#     cos = np.cos(angle)
#     sin = np.sin(angle)
#     env = torch.stack([
#         env[0] + env[6]*c5*cos*cos/c4 - env[6]*c5/c4 - 2*env[7]*c1*c5*sin*cos/(c3*c4) + env[8]*c1*c5*sin*sin/(c3*c4),
#         env[1],
#         env[2]*cos - env[3]*sin,
#         env[2]*sin + env[3]*cos,
#         env[4]*cos + env[5]*sin,
#         -env[4]*sin + env[5]*cos,
#         env[6]*cos*cos - 2*env[7]*c1*sin*cos/c3 + env[8]*c1*sin*sin/c3,
#         env[6]*c3*sin*cos/c1 - env[7]*sin*sin + env[7]*cos*cos - env[8]*sin*cos,
#         env[6]*c3*sin*sin/c1 + 2*env[7]*sin*cos + env[8]*cos*cos], 0)
#     return env

# def rotate_env(env, angle):
#     n = env.new_empty((30, 3)).normal_()
#     n = n / torch.norm(n, 2)
#     nr = rotate_vec(n, -angle)
#
#     newenv = nn.Parameter(env.clone())
#     # newenv = nn.Parameter(torch.randn_like(env.new_empty(9, 3)))
#
#     # opt = optim.Adam([newenv], lr=1)
#     opt = optim.LBFGS([newenv])
#
#     orill = illuminate_vec(n, env)
#
#     for it in range(10):
#         def closure():
#             opt.zero_grad()
#             newill = illuminate_vec(nr, newenv)
#             loss = torch.mean((newill-orill)**2)
#             loss.backward()
#             return loss
#         opt.step(closure)
#         # print(closure())
#     return newenv.detach()

if __name__ == '__main__':
    env = torch.randn(9, 3)
    angle = np.pi/2
    print(env)
    print(rotate_env(env, angle))
    # import timeit
    # # print(timeit.timeit(lambda: rotate_env(env, angle), number=100))
    # t = timeit.Timer(lambda: rotate_env(env, angle))
    # print((lambda c, t: t/c)(*t.autorange()))

