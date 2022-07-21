import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
# import mcubes
import marching_cubes as mcubes
import logging
from tqdm import tqdm, trange
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, create_nerf

logger = logging.getLogger(__package__)

def ddp_mesh_nerf(rank, args):
    ###### set up multi-processing
    assert(args.world_size==1)
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    # center on lk
    ax = np.linspace(-1, 1, num=1000, endpoint=True, dtype=np.float16)/5
    X, Y, Z = np.meshgrid(ax, ax-0.1, ax-0.3)

    # flip yz
    pts = np.stack((X, Y, Z), -1)
    pts = pts.reshape((-1, 3))

    pts = torch.tensor(pts).float().to(rank)

    u = models['net_1']
    nerf_net = u.module.nerf_net
    fg_net = nerf_net.fg_net

    allres = []
    allcolor = []
    with torch.no_grad():
        posemb = nerf_net.fg_embedder_position
        vdemb = nerf_net.fg_embedder_viewdir
        # direction = torch.tensor([0, 0, -1], dtype=torch.float32).to(rank)
        for bid in trange((pts.shape[0]+args.chunk_size-1)//args.chunk_size):
            bstart = bid * args.chunk_size
            bend = bstart + args.chunk_size
            cpts = pts[bstart:bend].float()
            cem = (cpts[..., 0:1]*0).expand((cpts.shape[0], 9))
            cvd = cpts*0#+direction

            inp = torch.cat((posemb(cpts, start), cem, vdemb(cvd, start)), -1)

            out = fg_net(inp)

            res = out['sigma'].detach().cpu().half().numpy()
            allres.append(res)
            color = out['rgb'].detach().cpu().half().numpy()
            allcolor.append(color)

    allres = np.concatenate(allres, 0)
    allres = allres.reshape(X.shape)

    allcolor = np.concatenate(allcolor, 0)
    allcolor = allcolor.reshape(list(X.shape)+[3,])

    print(allres.min(), allres.max(), allres.mean(), np.median(allres), allres.shape)

    logger.info('Doing MC')
    # vtx, tri = mcubes.marching_cubes(allres.astype(np.float32), 100)
    vtx, tri = mcubes.marching_cubes_color(allres.astype(np.float32), allcolor.astype(np.float32), 200)
    logger.info('Exporting mesh')
    # mcubes.export_mesh(vtx, tri, "mesh5.dae", "Mesh")
    mcubes.export_obj(vtx, tri, "meshcolor1000_01_03_t200.obj")


def mesh():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_mesh_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    mesh()

