import warnings
import time
import os
import pickle
import shutil

import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl._utils import logger as torchrl_logger

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import make_collector, make_replay_buffer, save_checkpoint, load_checkpoint
from utils_ddpg import (
    log_metrics,
    make_environment,
    make_loss_module,
    make_ddpg_agent,
    make_ddpg_optimizer,
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
    parser = argparse.ArgumentParser(description="DDPG Training with Checkpointing")
    parser.add_argument("--config_file", help="Configuration file path", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

    args = parser.parse_args()

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

        cfg = omegaconf.OmegaConf.load(args.config_file)
        print('The following configuration has been submitted and loaded:', flush=True)
        print(cfg, flush=True)
        print('----------------------------------------------', flush=True)

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

        train_env = make_MAIA_FlowEnv_torchrl(environment=cfg.maia.environment,
                                            env_config=env_config)

        train_env.set_seed(cfg.env.seed)
        train_env = apply_env_transforms(train_env, max_episode_steps=cfg.env.max_episode_steps)

        # Create agent
        model, exploration_policy = make_ddpg_agent(cfg, train_env, train_env, device)

        # Create DDPG loss
        loss_module, target_net_updater = make_loss_module(cfg, model)

        critic_params = list(loss_module.value_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())

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
        ) = make_ddpg_optimizer(cfg, loss_module)

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

        init_random_frames = cfg.collector.init_random_frames
        num_updates = int(
            cfg.collector.env_per_collector
            * cfg.collector.frames_per_batch
            * cfg.optim.utd_ratio
        )
        delayed_updates = cfg.optim.policy_update_delay
        prb = cfg.replay_buffer.prb
        eval_iter = cfg.logger.eval_iter
        frames_per_batch = cfg.collector.frames_per_batch
        eval_rollout_steps = cfg.env.max_episode_steps_validation

        sampling_start = time.time()

        collector_iter = iter(collector)
        total_iter = len(collector)

        for iteration in range(total_iter):
            # Check if we've reached the target frames (important for resumed training)
            if collected_frames >= total_frames:
                print(f"\n✓ Reached target frames ({total_frames}), stopping training")
                break
                
            sampling_time = time.time() - sampling_start

            tensordict = next(collector_iter)
            # Update exploration policy
            exploration_policy[1].step(tensordict.numel())

            # Update weights of the inference policy
            collector.update_policy_weights_()

            pbar.update(tensordict.numel())

            tensordict = tensordict.reshape(-1)
            current_frames = tensordict.numel()
            # Add to replay buffer
            replay_buffer.extend(tensordict.cpu())
            collected_frames += current_frames

            # Save checkpoint periodically
            if (cfg.env.save_checkpoint_every > 0 and 
                collected_frames > 0 and 
                collected_frames % cfg.env.save_checkpoint_every == 0):
                print(f"\n💾 Saving periodic checkpoint at frame {collected_frames}")
                save_checkpoint(cfg, model, optimizer_actor, optimizer_critic, replay_buffer,
                               collected_frames, update_counter, highest_reward_tracking,
                               exploration_policy, checkpoint_type="periodic")

            # Optimization steps
            training_start = time.time()
            if collected_frames >= init_random_frames:
                (
                    actor_losses,
                    q_losses,
                ) = ([], [])
                for _ in range(num_updates):

                    # Update actor every delayed_updates
                    update_counter += 1
                    update_actor = update_counter % delayed_updates == 0

                    # Sample from replay buffer
                    sampled_tensordict = replay_buffer.sample()
                    if sampled_tensordict.device != device:
                        sampled_tensordict = sampled_tensordict.to(
                            device, non_blocking=True
                        )
                    else:
                        sampled_tensordict = sampled_tensordict.clone()

                    # Compute loss
                    q_loss, *_ = loss_module.loss_value(sampled_tensordict)

                    # Update critic
                    optimizer_critic.zero_grad()
                    q_loss.backward()
                    torch.nn.utils.clip_grad_value_(critic_params, clip_value=10000)
                    optimizer_critic.step()
                    q_losses.append(q_loss.item())

                    # Update actor
                    if update_actor:
                        actor_loss, *_ = loss_module.loss_actor(sampled_tensordict)
                        optimizer_actor.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_value_(actor_params, clip_value=10000)
                        optimizer_actor.step()

                        actor_losses.append(actor_loss.item())

                        # Update target params
                        target_net_updater.step()

                    # Update priority
                    if prb:
                        replay_buffer.update_priority(sampled_tensordict)

            training_time = time.time() - training_start
            episode_rewards = tensordict["next", "reward"].sum(-2)

            # Logging
            metrics_to_log = {}
            if len(episode_rewards) > 0:
                metrics_to_log["train/reward"] = episode_rewards.mean().item()

            if collected_frames >= init_random_frames:
                metrics_to_log["train/q_loss"] = np.mean(q_losses)
                if update_actor:
                    metrics_to_log["train/a_loss"] = np.mean(actor_losses)
                metrics_to_log["train/sampling_time"] = sampling_time
                metrics_to_log["train/training_time"] = training_time

            # Evaluation
            if abs(collected_frames % eval_iter) < frames_per_batch:
                with set_exploration_type(ExplorationType.MODE), torch.no_grad():
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
        
        # Save final checkpoint
        print("\n💾 Saving final checkpoint")
        save_checkpoint(cfg, model, optimizer_actor, optimizer_critic, replay_buffer,
                       collected_frames, update_counter, highest_reward_tracking,
                       target_net_updater, exploration_policy, checkpoint_type="latest")

        collector.shutdown()


if __name__ == "__main__":
    main()