#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import yaml
import random
import numpy as np
from dotmap import DotMap
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config


import aihabitat.rl.ppo_agent.ppo_trainer as ppo_trainer
import aihabitat.rl.ppo_agent.transfer_ppo_trainer as transfer_ppo
import aihabitat.rl.random_agent as random_agent

def train_agent(agent):
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config_file = r"/home/userone/workspace/bed1/ppo_custom/ppo_pointnav_example.yaml"
    config = get_config(config_file, None)
    # config = get_config1(agent)
    env = construct_envs(config, get_env_class(config.ENV_NAME))

    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = config.DATA_PATH_SET
    config.TASK_CONFIG.DATASET.SCENES_DIR = config.SCENE_DIR_SET
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    if agent == "ppo_agent":
        trainer = ppo_trainer.PPOTrainer(config)

    trainer.train(env)


def test_agent(agent):
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config_file = r"/home/userone/workspace/bed1/ppo_custom/ppo_pointnav_example.yaml"
    checkpoint_file = r"/home/userone/workspace/bed1/ppo_custom/tmp/ppo_transfer_rgb/new_checkpoints/ckpt.19.pth"

    # config = get_config1(agent)
    config = get_config(config_file, None)
    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = config.DATA_PATH_SET
    config.TASK_CONFIG.DATASET.SCENES_DIR = config.SCENE_DIR_SET
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    if agent == "ppo_agent":
        trainer = ppo_trainer.PPOTrainer(config)

    elif agent == "ppo_rgb_transfer":
        trainer = transfer_ppo.PPOTrainer(config)


    trainer.eval(checkpoint_file)

def trigger_transfer_learn(agent):
    config_file = r"/home/userone/workspace/bed1/ppo_custom/ppo_pointnav_example.yaml"
    config = get_config(config_file, None)
    # config = get_config1(agent)
    env = construct_envs(config, get_env_class(config.ENV_NAME))

    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = config.DATA_PATH_SET
    config.TASK_CONFIG.DATASET.SCENES_DIR = config.SCENE_DIR_SET
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    if agent == "ppo_agent":
        trainer = transfer_ppo.PPOTrainer(config)

    trainer.train(env)

def run_random_agent():
    config_file = r"/home/userone/workspace/bed1/ppo_custom/ppo_pointnav_example.yaml"
    config = get_config(config_file, None)
    # config = get_config1(agent)
    env = construct_envs(config, get_env_class(config.ENV_NAME))

    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = config.DATA_PATH_SET
    config.TASK_CONFIG.DATASET.SCENES_DIR = config.SCENE_DIR_SET
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)


    random_agent.run(config, env, 3)