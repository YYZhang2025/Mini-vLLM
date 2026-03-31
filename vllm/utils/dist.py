import torch.distributed as dist


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_dist(backend="nccl", **kwargs):
    dist.init_process_group(backend=backend, **kwargs)
