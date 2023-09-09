import os
import hydra
import numpy as np
import joblib
import torch
import wandb
from helpers.experiment import (
    setup_wandb,
    hydra_to_dict,
    flatten_dict,
    set_random_seed,
)
from common.utils import set_gpu_mode, get_device

from ppo import PPO
from environments import load_envs
from common.logger import configure_logger


@hydra.main(config_path="configs/", config_name="reach", version_base=None)
def run_experiment(cfg):

    # # if using wandb
    #if "wandb" in cfg.log.format_strings:
    run = setup_wandb(cfg, name=f"{cfg.exp_id}[reach][{cfg.seed}]")
    # if not using wandb
    #cfg.log.format_strings = ["stdout", "tensorboard"]
    
    set_random_seed(cfg.seed)
    set_gpu_mode(cfg.gpu_id>=0, gpu_id=cfg.gpu_id)
    device = get_device() #torch.device("cuda:0")  

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "task")
    logger = configure_logger(logdir, cfg.log.format_strings)

    # sim
    cfg_env_dict = flatten_dict(hydra_to_dict(cfg.env))
    print(cfg_env_dict)
    envs = load_envs(cfg_env_dict, num_workers=cfg.num_workers, seed=cfg.seed)
    eval_envs = load_envs(
        cfg_env_dict, num_workers=cfg.num_workers, seed=cfg.seed, eval=True
    )

    envs.reset()
    eval_envs.reset()

    # Agent
    algo_fn = globals()[cfg.train.algorithm.algo_name.upper()]  
    cfg.train.algorithm.eval_freq = cfg.log.eval_interval
    cfg.train.algorithm.checkpoint_freq = cfg.log.save_interval
    cfg.train.algorithm.num_envs = cfg.num_workers
    algo = algo_fn(cfg.train.algorithm, envs, eval_envs, logger, device)

    # never train on real robot
    if cfg.env.domain.sim:
        algo.train()

    algo.load_checkpoint()
    algo.evaluate()

    # # always dump real robot data
    # if not cfg.env.domain.sim:
    #     eval_envs.dump_data(f"{cfg.env_id}")

    envs.close()


if __name__ == "__main__":
    run_experiment()
