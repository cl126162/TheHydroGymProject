import warnings


import time
import os

import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl._utils import logger as torchrl_logger

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import make_collector, make_replay_buffer, save_checkpoint, load_checkpoint
from utils_ppo import (
    log_metrics,
    make_loss_module,
    make_ppo_agent,
    make_ppo_optimizer,
)

from maiaGym_TorchRL_wrapper import make_MAIA_FlowEnv_torchrl, apply_env_transforms
from torchrl.envs.transforms import ObservationNorm
from torchrl.envs import TransformedEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import omegaconf


def main():  # noqa: F821
    parser = argparse.ArgumentParser(description="My Awesome Script")
    parser.add_argument("--config_file", help="Property File")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

    args = parser.parse_args()

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        cfg = omegaconf.OmegaConf.load(args.config_file)
        device = torch.device(cfg.network.device)

        # Create logger
        logger = None
        if cfg.logger.backend:
            logger = get_logger(
                logger_type=cfg.logger.backend,
                logger_name=cfg.env.data_path + "tensorboard",
                experiment_name="",
                wandb_kwargs={"mode": cfg.logger.mode, "config": cfg},
            )

        torch.manual_seed(cfg.env.seed)
        np.random.seed(cfg.env.seed)

        # --- Generate Probe Locations ---
        # Define ranges for x/y dimensions
        xMin, xMax, nXProbes = 2.0, 8, 7
        yMin, yMax, nYProbes = -0.75, 0.75, 5
        # Create 1D arrays for each dimension
        xp = np.linspace(xMin, xMax, nXProbes)
        yp = np.linspace(yMin, yMax, nYProbes)

        if cfg.maia.nDim == 2:
            # Create 2D meshgrid
            X, Y = np.meshgrid(xp, yp)
            # Create list of (x, y, z) tuples
            probe_list = probe_list + [(x, y) for x, y in zip(X.ravel(), Y.ravel())]
            # Flatten to a single list
            probe_locations = [item for sublist in probe_list for item in sublist]
        else:
            zMin, zMax, nZProbes = -1.5, 1.5, 5  # Add Z dimension
            zp = np.linspace(zMin, zMax, nZProbes)
            # Create 3D meshgrid
            X, Y, Z = np.meshgrid(xp, yp, zp)
            # Create list of (x, y, z) tuples
            probe_list = [(x, y, z) for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())]
            # Flatten to a single list
            probe_locations = [item for sublist in probe_list for item in sublist]
        
        nDim = getattr(cfg.maia, 'nDim', 2)
        obs_loc = [0.0] * int(len(probe_locations) // nDim) * len(cfg.maia.observation_type)
        obs_scale = [1.0] * int(len(probe_locations) // nDim) * len(cfg.maia.observation_type)

        # only cavity
        reward_loc = [0.0] * int(len(probe_locations) // nDim) * len(cfg.maia.observation_type)
        reward_scale = [1.0] * int(len(probe_locations) // nDim) * len(cfg.maia.observation_type)

        
        env_config = {
            "configuration_file": args.config_file,
            "is_testing": False,
            "render": cfg.maia.render,
            # set customized probes values
            "probe_locations": probe_locations,
            # if applicable, set normalization values via obs_loc and obs_scale - otherwise will be computed
            "obs_loc": obs_loc,
            "obs_scale": obs_scale,
            # reward scale - only cavity
            "reward_loc": reward_loc,
            "reward_scale": reward_scale,
            }

        # create TorchRL environment
        train_env = make_MAIA_FlowEnv_torchrl(environment=cfg.maia.environment,
                                            env_config=env_config)
        train_env.set_seed(cfg.env.seed)
        train_env = apply_env_transforms(train_env, max_episode_steps=cfg.env.max_episode_steps)

        # Create agent
        model, exploration_policy = make_ppo_agent(cfg, train_env, train_env, device)

        # Create PPO loss
        adv_module, loss_module = make_loss_module(cfg, model)

        # Create off-policy collector
        collector = make_collector(cfg, train_env, exploration_policy)

        # Create replay buffer
        replay_buffer = make_replay_buffer(
            batch_size=cfg.optim.batch_size,
            prb=cfg.replay_buffer.prb,
            buffer_size=cfg.replay_buffer.size,
            device="cpu",
            agent=cfg.env.agent
        )

        # Create optimizers
        (
            optimizer_actor,
            optimizer_critic,
        ) = make_ppo_optimizer(cfg, loss_module)

        # Initialize training state
        collected_frames = 0
        update_counter = 0
        
        # Try to resume from checkpoint if requested
        if args.resume:
            print("Attempting to resume from checkpoint...")
            loaded_frames, loaded_counter, loaded_rewards, loaded_buffer = load_checkpoint(
                cfg, model, optimizer_actor, optimizer_critic, exploration_policy, device
            )
            
            if loaded_frames is not None:
                collected_frames = loaded_frames
                update_counter = loaded_counter
                highest_reward_tracking = loaded_rewards
                
                if loaded_buffer is not None:
                    replay_buffer = loaded_buffer
                
                print(f"✓ Successfully resumed training from frame {collected_frames}")
                
                # Update collector policy weights to match loaded model
                collector.update_policy_weights_()
            else:
                print("⚠ Could not load checkpoint, starting fresh training")
        else:
            print("Starting fresh training (use --resume to resume from checkpoint)")


        # Main loop
        start_time = time.time()
        
        # Adjust progress bar to account for resumed training
        total_frames = cfg.collector.total_frames
        remaining_frames = max(0, total_frames - collected_frames)
        pbar = tqdm.tqdm(total=remaining_frames, desc="Training Progress")

        num_mini_batches = cfg.collector.frames_per_batch // cfg.optim.batch_size
        total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.optim.ppo_epochs
            * num_mini_batches
        )

        sampling_start = time.time()

        # extract cfg variables
        cfg_loss_ppo_epochs = cfg.optim.ppo_epochs
        cfg_optim_anneal_lr = cfg.optim.anneal_lr
        cfg_actor_lr = cfg.optim.actor_lr
        cfg_critic_lr = cfg.optim.critic_lr
        cfg_loss_anneal_clip_eps = cfg.optim.anneal_clip_epsilon
        cfg_loss_clip_epsilon = cfg.optim.clip_epsilon

        prb = cfg.replay_buffer.prb
        init_random_frames = cfg.collector.init_random_frames
        eval_iter = cfg.logger.eval_iter
        frames_per_batch = cfg.collector.frames_per_batch
        eval_rollout_steps = cfg.env.max_episode_steps_validation

        losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

        for i, data in enumerate(collector):

            log_info = {}
            sampling_time = time.time() - sampling_start
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(data.numel())

            episode_rewards = data["next", "reward"].sum(-2)
            if len(episode_rewards) > 0:
                log_info.update(
                    {
                        "train/reward": episode_rewards.mean().item(),
                    }
                )
            

            training_start = time.time()
            for j in range(cfg_loss_ppo_epochs):

                # Compute GAE
                with torch.no_grad():
                    data = adv_module(data)
                data_reshape = data.reshape(-1)

                # Update the data buffer
                replay_buffer.extend(data_reshape)

                for k, batch in enumerate(replay_buffer):

                    # Get a data batch
                    batch = batch.to(device)

                    # Linearly decrease the learning rate and clip epsilon
                    alpha = 1.0
                    if cfg_optim_anneal_lr:
                        alpha = 1 - (update_counter / total_network_updates)
                        for group in optimizer_actor.param_groups:
                            group["lr"] = cfg_actor_lr * alpha
                        for group in optimizer_critic.param_groups:
                            group["lr"] = cfg_critic_lr * alpha
                    if cfg_loss_anneal_clip_eps:
                        loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                    update_counter += 1

                    # Forward pass PPO loss
                    loss = loss_module(batch)
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    ).detach()
                    critic_loss = loss["loss_critic"]
                    actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                    # Backward pass
                    actor_loss.backward()
                    critic_loss.backward()

                    # Update the networks
                    optimizer_actor.step()
                    optimizer_critic.step()
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()

            # Get training losses and times
            training_time = time.time() - training_start
            losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses_mean.items():
                log_info.update({f"train/{key}": value.item()})

            # Logging
            metrics_to_log = {}
            if len(episode_rewards) > 0:
                metrics_to_log["train/reward"] = episode_rewards.mean().item()

            if collected_frames >= init_random_frames:
                metrics_to_log["train/loss_critic"] = losses_mean["loss_critic"]
                metrics_to_log["train/actor_loss"] = losses_mean["loss_objective"] + losses_mean["loss_entropy"]
                metrics_to_log["train/actor_lr"] = alpha * cfg_actor_lr
                metrics_to_log["train/critic_lr"] = alpha * cfg_critic_lr
                metrics_to_log["train/clip_epsilon"] = alpha * cfg_loss_clip_epsilon if cfg_loss_anneal_clip_eps else cfg_loss_clip_epsilon
                metrics_to_log["train/entropy"] = losses_mean["loss_entropy"]
                metrics_to_log["train/sampling_time"] = sampling_time
                metrics_to_log["train/training_time"] = training_time

            # Evaluation
            if abs(collected_frames % eval_iter) < frames_per_batch:
                with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                    eval_start = time.time()
                    eval_rollout = train_env.rollout(
                        eval_rollout_steps,
                        model[0],
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    eval_time = time.time() - eval_start
                    eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                    metrics_to_log["eval/reward"] = eval_reward
                    metrics_to_log["eval/time"] = eval_time

                ckpt_dir = os.path.join(cfg.env.data_path, 'checkpoints')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                # Save latest training checkpoint during evaluation
                save_checkpoint(cfg, model, optimizer_actor, optimizer_critic, replay_buffer,
                               collected_frames, update_counter, highest_reward_tracking,
                               exploration_policy, checkpoint_type="latest")

            if logger is not None:
                log_metrics(logger, metrics_to_log, collected_frames)
            sampling_start = time.time()


        collector.shutdown()


if __name__ == "__main__":
    main()