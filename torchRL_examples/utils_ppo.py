# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict.nn import InteractionType, TensorDictModule, AddStateIndependentNormalScale
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, CompositeSpec
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

# ====================================================================
# Collector and replay buffer
# ---------------------------

def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        # max_frames_per_traj=cfg.collector.max_episode_steps,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    device="cpu",
    prefetch=3,
):
    sampler = SamplerWithoutReplacement()
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                device=device,
            ),
            sampler=sampler,
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                device=device,
            ),
            sampler=sampler,
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----

def make_ppo_agent(cfg, train_env, eval_env, device):
    """Make PPO agent."""

    # Define input shape
    # input_shape = train_env.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = train_env.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": train_env.action_spec.space.low,
        "high": train_env.action_spec.space.high,
        "tanh_loc": False,
    }

    
    # Define policy architecture
    policy_mlp = MLP(
        # in_features=input_shape[-1],
        activation_class=get_activation(cfg),
        out_features= 2 * num_outputs,  
        num_cells=cfg.network.hidden_sizes,
    )

    actor_extractor = NormalParamExtractor()
    actor_net = nn.Sequential(policy_mlp, actor_extractor)

    policy_module = TensorDictModule(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
    
    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        # spec=CompositeSpec(action=train_env.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    value_module = ValueOperator(
        in_keys=["observation"],
        module=qvalue_net,
    )

    model = nn.ModuleList([policy_module, value_module]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0]

# ====================================================================
# PPO Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.optim.gamma,
        lmbda=cfg.optim.gae_lambda,
        value_network=model[1],
        average_gae=cfg.optim.average_gae,
    )

    loss_module = ClipPPOLoss(
        actor_network=model[0],
        critic_network=model[1],
        clip_epsilon=cfg.optim.clip_epsilon,
        loss_critic_type=cfg.optim.loss_critic_type,
        entropy_coef=cfg.optim.entropy_coef,
        critic_coef=cfg.optim.critic_coef,
        normalize_advantage=cfg.optim.normalize_advantage,
    )

    return adv_module, loss_module


def make_ppo_optimizer(cfg, loss_module):
    critic_params = list(loss_module.critic_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.actor_lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.critic_lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )

    return optimizer_actor, optimizer_critic


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    elif cfg.network.activation == "mish":
        return nn.Mish
    else:
        raise NotImplementedError