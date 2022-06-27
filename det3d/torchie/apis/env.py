import logging
import os
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from det3d.torchie.trainer import get_dist_info


def init_dist(launcher, backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == "mpi":
        _init_dist_mpi(backend, **kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError("Invalid launcher type: {}".format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError


def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["RANK"] = str(proc_id)
    dist.init_process_group(backend=backend)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_file, log_level=logging.INFO):
    logger = logging.getLogger()
    # if not logger.hasHandlers():
    #     logging.basicConfig(
    #         format="%(asctime)s - %(levelname)s - %(message)s", level=log_level
    #     )
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def clever_format(nums, format="%.2f"):
    if nums > 1e12:
        return format % (nums / 1e12) + "T"
    elif nums > 1e9:
        return format % (nums / 1e9) + "G"
    elif nums > 1e6:
        return format % (nums / 1e6) + "M"
    elif nums > 1e3:
        return format % (nums / 1e3) + "K"
    else:
        return format % (nums) + "B"


def get_model_params(model):
    total_params = 0
    total_trainable = 0
    for m in model.parameters():
        params, t_params = 0, 0
        params += m.numel()
        if m.requires_grad:
            t_params += m.numel()
        total_params += params
        total_trainable += t_params

    print("total network parameters: ", clever_format(total_params))
    print("total trainable parameters: ", clever_format(total_trainable))
