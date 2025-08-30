import torch.distributed as dist

def dist_inited():
    return dist.is_available() and dist.is_initialized()

def barrier_all():
    if dist_inited():
        dist.barrier()