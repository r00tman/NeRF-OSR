import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from nerf_network import Embedder, MLPNet
from sph_util import illuminate_vec, rotate_env
import logging

logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2,
                                             N_anneal=args.N_anneal,
                                             N_anneal_min_freq=args.N_anneal_min_freq,
                                             use_annealing=args.use_annealing)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs,
                                            N_anneal=args.N_anneal,
                                            N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
                                            use_annealing=args.use_annealing)
        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             use_shadow=True,
                             act=args.activation)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2,
                                             N_anneal=args.N_anneal,
                                             N_anneal_min_freq=args.N_anneal_min_freq,
                                             use_annealing=args.use_annealing)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs,
                                            N_anneal=args.N_anneal,
                                            N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
                                            use_annealing=args.use_annealing)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             use_shadow=False,
                             act=args.activation)

        self.with_bg = args.with_bg

        self.use_shadow_jitter = args.use_shadow_jitter
        self.use_shadows = args.use_shadows

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, env, iteration):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        env_gray = env[..., 0]*0.2126 + env[..., 1]*0.7152 + env[..., 2]*0.0722
        fg_sph = env_gray.view(9).unsqueeze(0).unsqueeze(0).expand(dots_sh + [N_samples, 9])

        if self.use_shadow_jitter:
            fg_sph = fg_sph + torch.randn_like(fg_sph)*0.01

        # fg_viewdirs = fg_viewdirs * 0  # todo: disable viewdirs, because we need albedo
        with torch.enable_grad():
            fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
            fg_pts.requires_grad_(True)
            input = torch.cat((self.fg_embedder_position(fg_pts, iteration),
                               fg_sph,
                               self.fg_embedder_viewdir(fg_viewdirs, iteration)), dim=-1)
            fg_raw = self.fg_net(input)
            # sigmamasked = fg_raw['sigma']*(fg_raw['sigma'] < 4.0)
            # fg_raw['sigma'] = fg_raw['sigma'] - sigmamasked
            fg_normal_map = torch.autograd.grad(
                outputs=fg_raw['sigma'],
                inputs=fg_pts,
                grad_outputs=torch.ones_like(fg_raw['sigma'], requires_grad=False),
                retain_graph=True,
                create_graph=True)[0]
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        fg_albedo_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_shadow_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['shadow'], dim=-2)  # [..., 3]

        if not self.use_shadows:
            fg_shadow_map = fg_shadow_map * 0 + 1

        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)  # [...,]
        # print(fg_pts.shape, fg_depth_map.shape, fg_raw['sigma'].shape)
        fg_normal_map = (fg_normal_map * fg_weights.unsqueeze(-1)).mean(-2)
        # fg_normal_map = fg_normal_map.mean(-2)
        fg_normal_map = F.normalize(fg_normal_map, p=2, dim=-1)
        # print(fg_normal_map.shape)

        # c1 = 0.429043
        # c2 = 0.511664
        # c3 = 0.743125
        # c4 = 0.886227
        # c5 = 0.247708
        # c = env.unsqueeze(1)
        # n = fg_normal_map
        # def rotate_xz(v, rot_angle):
        #     mat = v.new_zeros((3, 3))
        #     cos = np.cos(rot_angle)
        #     sin = np.sin(rot_angle)
        #     mat[0,0] = cos
        #     mat[0,2] = -sin
        #     mat[2,0] = sin
        #     mat[2,2] = cos
        #     return v @ mat.T
        # cos = np.cos(rot_angle)
        # sin = np.sin(rot_angle)

        # n = rotate_xz(n, rot_angle)
        # irradiance = (
        #     c4 * c[0] - c5 * c[6] +
        #     n[..., 0, None] * (2 * c2 * sin * c[2] + 2 * c2 * cos * c[3]) +
        #     n[..., 1, None] * (2 * c2 * c[1]) +
        #     n[..., 2, None] * (2 * c2 * cos * c[2] - 2 * c2 * sin * c[3]) +
        #     (n[..., 0, None] ** 2) * (c3*sin*sin*c[6]+2*c1*sin*cos*c[7]+c1*cos*cos*c[8]) +
        #     (n[..., 1, None] ** 2) * (-c1 * c[8]) +
        #     (n[..., 2, None] ** 2) * (c3*cos*cos*c[6]-2*c1*sin*cos*c[7]+c1*sin*sin*c[8]) +
        #     n[..., 0, None] * n[..., 1, None] * (2*c1*cos*c[4]+2*c1*sin*c[5]) +
        #     n[..., 0, None] * n[..., 2, None] * (2*c3*sin*cos*c[6]+2*c1*(cos*cos-sin*sin)*c[7]-2*c1*sin*cos*c[8]) +
        #     n[..., 1, None] * n[..., 2, None] * (-2*c1*sin*c[4]+2*c1*cos*c[5])
        # )

        # irradiance = (
        #         c1 * c[8] * (n[..., 0, None] ** 2 - n[..., 1, None] ** 2) +
        #         c3 * c[6] * (n[..., 2, None] ** 2) +
        #         c4 * c[0] -
        #         c5 * c[6] +
        #         2 * c1 * c[4] * n[..., 0, None] * n[..., 1, None] +
        #         2 * c1 * c[7] * n[..., 0, None] * n[..., 2, None] +
        #         2 * c1 * c[5] * n[..., 1, None] * n[..., 2, None] +
        #         2 * c2 * c[3] * n[..., 0, None] +
        #         2 * c2 * c[1] * n[..., 1, None] +
        #         2 * c2 * c[2] * n[..., 2, None]
        # )
        irradiance = illuminate_vec(fg_normal_map, env)
        irradiance = torch.relu(irradiance)  # can't be < 0
        irradiance = irradiance ** (1 / 2.2)  # linear to srgb
        fg_pure_rgb_map = irradiance * fg_albedo_map
        fg_rgb_map = fg_pure_rgb_map * fg_shadow_map

        # render background
        if self.with_bg:
            N_samples = bg_z_vals.shape[-1]
            bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
            bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
            bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
            bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
            input = torch.cat((self.bg_embedder_position(bg_pts, iteration),
                               self.bg_embedder_viewdir(bg_viewdirs, iteration)), dim=-1)
            # near_depth: physical far; far_depth: physical near
            input = torch.flip(input, dims=[-2, ])
            bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
            bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
            bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
            bg_raw = self.bg_net(input)
            bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
            # Eq. (3): T
            # maths show weights, and summation of weights along a ray, are always inside [0, 1]
            T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
            T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
            bg_weights = bg_alpha * T  # [..., N_samples]

            bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
            bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

            # composite foreground and background
            bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
            bg_depth_map = bg_lambda * bg_depth_map
        else:
            bg_rgb_map = fg_rgb_map*0
            bg_depth_map = fg_depth_map*0
            bg_weights = fg_weights*0

        if self.with_bg:
            pure_rgb_map = fg_pure_rgb_map + bg_rgb_map
            shadow_map = fg_shadow_map
            rgb_map = fg_rgb_map + bg_rgb_map  # todo: better compose fg
        else:
            pure_rgb_map = fg_pure_rgb_map + bg_rgb_map * 0
            shadow_map = fg_shadow_map
            rgb_map = fg_rgb_map + bg_rgb_map * 0  # todo: enable bg later

        ret = OrderedDict([('rgb', rgb_map),  # loss
                           ('pure_rgb', pure_rgb_map),
                           ('shadow', shadow_map),
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),  # below are for logging
                           ('fg_albedo', fg_albedo_map.detach()),
                           ('fg_shadow', fg_shadow_map.detach()),
                           ('fg_depth', fg_depth_map.detach()),
                           ('fg_normal', fg_normal_map.detach()),
                           ('irradiance', irradiance.detach()),
                           ('bg_rgb', bg_rgb_map.detach()),
                           ('bg_depth', bg_depth_map.detach()),
                           ('bg_lambda', bg_lambda.detach()),
                           ('viewdir', viewdirs.detach())])
        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.test_env = args.test_env

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

        assert (img_names is not None)
        logger.info('Optimizing envmap!')

        self.img_names = [remap_name(x) for x in img_names]
        logger.info('\n'.join(self.img_names))
        self.env_params = nn.ParameterDict(OrderedDict(
            [(x, nn.Parameter(torch.tensor([
                [2.9861e+00, 3.4646e+00, 3.9559e+00],
                [1.0013e-01, -6.7589e-02, -3.1161e-01],
                [-8.2520e-01, -5.2738e-01, -9.7385e-02],
                [2.2311e-03, 4.3553e-03, 4.9501e-03],
                [-6.4355e-03, 9.7476e-03, -2.3863e-02],
                [1.1078e-01, -6.0607e-02, -1.9541e-01],
                [7.9123e-01, 7.6916e-01, 5.6288e-01],
                [6.5793e-02, 4.3270e-02, -1.7002e-01],
                [-7.2674e-02, 4.5177e-02, 2.2858e-01]

                # [2.9861e+00, 3.4646e+00, 3.9559e+00],
                # [1.0013e-01, -6.7589e-02, -3.1161e-01],
                # [8.2520e-01, 5.2738e-01, 9.7385e-02],
                # [-2.2311e-03, -4.3553e-03, -4.9501e-03],
                # [6.4355e-03, -9.7476e-03, 2.3863e-02],
                # [-1.1078e-01, 6.0607e-02, 1.9541e-01],
                # [7.9123e-01, 7.6916e-01, 5.6288e-01],
                # [6.5793e-02, 4.3270e-02, -1.7002e-01],
                # [-7.2674e-02, 4.5177e-02, 2.2858e-01]
            ], dtype=torch.float32))) for x in self.img_names]))  # todo: limit to max 1
        self.register_buffer('defaultenv', torch.tensor([
                [2.9861e+00, 3.4646e+00, 3.9559e+00],
                [1.0013e-01, -6.7589e-02, -3.1161e-01],
                [-8.2520e-01, -5.2738e-01, -9.7385e-02],
                [2.2311e-03, 4.3553e-03, 4.9501e-03],
                [-6.4355e-03, 9.7476e-03, -2.3863e-02],
                [ 1.1078e-01, -6.0607e-02, -1.9541e-01],
                [7.9123e-01, 7.6916e-01, 5.6288e-01],
                [ 6.5793e-02,  4.3270e-02, -1.7002e-01],
                [-7.2674e-02, 4.5177e-02, 2.2858e-01]

                # [2.9861e+00, 3.4646e+00, 3.9559e+00],
                # [1.0013e-01, -6.7589e-02, -3.1161e-01],
                # [8.2520e-01, 5.2738e-01, 9.7385e-02],
                # [-2.2311e-03, -4.3553e-03, -4.9501e-03],
                # [6.4355e-03, -9.7476e-03, 2.3863e-02],
                # [-1.1078e-01, 6.0607e-02, 1.9541e-01],
                # [7.9123e-01, 7.6916e-01, 5.6288e-01],
                # [6.5793e-02, 4.3270e-02, -1.7002e-01],
                # [-7.2674e-02, 4.5177e-02, 2.2858e-01]
            # [1.3242, 1.2883, 1.2783],
            # [0.0256, 0.0296, 0.0315],
            # [0.0376, 0.0362, 0.0390],
            # [0.0057, 0.0016, 0.0027],
            # [-0.0066, -0.0036, -0.0015],
            # [-0.0329, -0.0395, -0.0416],
            # [-0.0350, -0.0316, -0.0352],
            # [0.0038, 0.0042, 0.0019],
            # [0.0124, 0.0130, 0.0108]

            # [0.7953949, 0.4405923, 0.5459412],
            # [0.3981450, 0.3526911, 0.6097158],
            # [-0.3424573, -0.1838151, -0.2715583],
            # [-0.2944621, -0.0560606, 0.0095193],
            # [-0.1123051, -0.0513088, -0.1232869],
            # [-0.2645007, -0.2257996, -0.4785847],
            # [-0.1569444, -0.0954703, -0.1485053],
            # [0.5646247, 0.2161586, 0.1402643],
            # [0.2137442, -0.0547578, -0.3061700]
        ], dtype=torch.float32))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, iteration, img_name=None, rot_angle=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        if img_name is not None:
            img_name = remap_name(img_name)
        env = None

        if self.test_env is not None:
            if not os.path.isdir(self.test_env):
                if 'test_env_val' not in dir(self):
                    env_data = np.loadtxt(self.test_env)
                    self.test_env_val = torch.tensor(env_data, dtype=torch.float32).to(ray_o.device)
                env = self.test_env_val
                logger.warning('using env ' + self.test_env)
            else:
                if 'test_env_val' not in dir(self):
                    self.test_env_val = dict()
                    for env_fn in sorted(glob.glob(os.path.join(self.test_env, '*'))):
                        env_data = np.loadtxt(env_fn)
                        env_name = os.path.splitext(os.path.basename(env_fn))[0]
                        self.test_env_val[env_name] = torch.tensor(env_data, dtype=torch.float32).to(ray_o.device)
                env_name = img_name.split('/')[-1][:-4]
                env = self.test_env_val[env_name]
                logger.warning('using env ' + env_name)

        elif img_name in self.env_params:
            env = self.env_params[img_name]
        else:
            logger.warning('no envmap found for ' + str(img_name))
            env = self.defaultenv
            # env = torch.tensor([
            #     [ 0.7953949,  0.4405923,  0.5459412],
            #     [ 0.3981450,  0.3526911,  0.6097158],
            #     [-0.3424573, -0.1838151, -0.2715583],
            #     [-0.2944621, -0.0560606,  0.0095193],
            #     [-0.1123051, -0.0513088, -0.1232869],
            #     [-0.2645007, -0.2257996, -0.4785847],
            #     [-0.1569444, -0.0954703, -0.1485053],
            #     [ 0.5646247,  0.2161586,  0.1402643],
            #     [ 0.2137442, -0.0547578, -0.3061700]
            #
            #     [1.3242, 1.2883, 1.2783],
            #     [0.0256, 0.0296, 0.0315],
            #     [0.0376, 0.0362, 0.0390],
            #     [0.0057, 0.0016, 0.0027],
            #     [-0.0066, -0.0036, -0.0015],
            #     [-0.0329, -0.0395, -0.0416],
            #     [-0.0350, -0.0316, -0.0352],
            #     [0.0038, 0.0042, 0.0019],
            #     [0.0124, 0.0130, 0.0108]
            # ]).to(ray_o.device)

        if rot_angle is not None:
            # c1 = 0.429043
            # c2 = 0.511664
            # c3 = 0.743125
            # c4 = 0.886227
            # c5 = 0.247708
            # cos = np.cos(rot_angle)
            # sin = np.sin(rot_angle)
            old_shape = env.shape
            env = rotate_env(env, rot_angle)
            # env = torch.stack([
            #     env[0] + env[6]*c5*cos*cos/c4 - env[6]*c5/c4 - 2*env[7]*c1*c5*sin*cos/(c3*c4) + env[8]*c1*c5*sin*sin/(c3*c4),
            #     env[1],
            #     env[2]*cos - env[3]*sin,
            #     env[2]*sin + env[3]*cos,
            #     env[4]*cos + env[5]*sin,
            #     -env[4]*sin + env[5]*cos,
            #     env[6]*cos*cos - 2*env[7]*c1*sin*cos/c3 + env[8]*c1*sin*sin/c3,
            #     env[6]*c3*sin*cos/c1 - env[7]*sin*sin + env[7]*cos*cos - env[8]*sin*cos,
            #     env[6]*c3*sin*sin/c1 + 2*env[7]*sin*cos + env[8]*cos*cos], 0)
            if env.shape != old_shape:
                print(env.shape, old_shape)
            env = env.reshape(old_shape)
            # assert(env.shape == old_shape)

        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, env, iteration)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
