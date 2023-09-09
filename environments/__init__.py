import os
import gym
from gym.wrappers import TimeLimit

from functools import partial

from .vec_env import VecEnv
from .franka.robot_env import RobotEnv

gym.logger.set_level(40)

def load_envs(cfg, seed=0, num_workers=1, device_id=0, eval=False):

    if cfg["sim"]:
        if num_workers > 1:
            envs = make_vec_env(cfg, num_workers=num_workers, seed=seed+100 if eval else seed, device_id=device_id)
        else:
            envs = make_env(cfg, seed=seed+100 if eval else seed, device_id=device_id, verbose=True)
    else:
        envs = make_env(cfg, seed=seed, device_id=device_id, verbose=True)
        envs.num_envs = 1
    
    return envs
    
def make_env(cfg, seed=0, device_id=0, verbose=False):
    env = RobotEnv(**cfg)
    env = TimeLimit(env, max_episode_steps=cfg["max_episode_steps"])
    env.seed(seed)
    if verbose:
        print(cfg)
    return env

def make_vec_env(cfg_dict, num_workers, seed, device_id=0):
    env_fns = [
        partial(
            make_env,
            cfg_dict,
            seed=seed + i,
            device_id=device_id,
            verbose=bool(i==0),
        )
        for i in range(num_workers)
    ]
    return VecEnv(env_fns)