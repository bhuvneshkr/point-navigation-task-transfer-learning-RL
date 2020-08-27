#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict, List, Optional

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from aihabitat.common.rollout_storage import RolloutStorage
from aihabitat.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)

from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.base_trainer import BaseRLTrainer
from aihabitat.rl.ppo_agent import ppo as ppo
from aihabitat.rl.ppo_agent import policy as policy


def run(config, env, max_steps):
    r"""Main method for training PPO.

    Returns:
        None
    """


    observations = env.reset()
    batch = batch_obs(observations)

    batch = None
    observations = None

    episode_rewards = torch.zeros(env.num_envs, 1)
    episode_counts = torch.zeros(env.num_envs, 1)
    episode_dist = torch.zeros(env.num_envs, 1)
    current_episode_reward = torch.zeros(env.num_envs, 1)


    window_episode_reward = deque(maxlen=max_steps)
    window_episode_counts = deque(maxlen=max_steps)
    dist_val = deque(maxlen=max_steps)

    t_start = time.time()
    env_time = 0
    pth_time = 0
    count_steps = 0
    count_checkpoints = 0

    for update in range(max_steps):
        print(update)
        reward_sum = 0
        dist_sum = 0
        iter = 0
        rgb_frames = []
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)


        # get name of performance metric, e.g. "spl"
        metric_name = config.TASK_CONFIG.TASK.MEASUREMENTS[0]
        metric_cfg = getattr(config.TASK_CONFIG.TASK, metric_name)
        measure_type = baseline_registry.get_measure(metric_cfg.TYPE)

        for step in range(500):
            dones = [False]
            while dones[0] == False:
                outputs = env.step([env.action_spaces[0].sample()])
                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
                reward_sum += rewards[0]
                dist_sum += observations[0]['pointgoal_with_gps_compass'][0]
                iter += 1

                frame = observations_to_image(observations[0], [])
                rgb_frames.append(frame)

        observations = env.reset()
        window_episode_reward.append(reward_sum/iter)
        window_episode_counts.append(iter)
        dist_val.append(dist_sum/iter)

        generate_video(
            video_option=config.VIDEO_OPTION,
            video_dir=config.VIDEO_DIR,
            images=np.array(rgb_frames),
            episode_id=update,
            checkpoint_idx=0,
            metric_name="spl",
            metric_value=1.0,
        )

        rgb_frames = []




    np.savetxt("window_episode_reward_ppo.csv", window_episode_reward, delimiter=",")
    np.savetxt("window_episode_counts_ppo.csv", window_episode_counts, delimiter=",")
    np.savetxt("episode_dist_ppo.csv", episode_dist, delimiter=",")

    env.close()

