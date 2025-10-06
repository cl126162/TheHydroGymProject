import os
import shutil
import torch
from tensordict.nn import InteractionType, TensorDictModule, AddStateIndependentNormalScale
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer, CompositeSpec
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

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
    agent='PPO'
):  
    if agent in ['PPO']:
        sampler = SamplerWithoutReplacement()
    else:
        sampler = None

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



def save_checkpoint(cfg, model, optimizer_actor, optimizer_critic, replay_buffer, 
                   collected_frames, update_counter, highest_reward_tracking, 
                   exploration_policy, checkpoint_type="latest"):
    """
    Save complete training checkpoint using TorchRL's native dumps() method
    """
    ckpt_dir = os.path.join(cfg.env.data_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if checkpoint_type == "latest":
        checkpoint_path = os.path.join(ckpt_dir, f'{cfg.env.configuration_name}_latest.pt')
        buffer_dir = os.path.join(ckpt_dir, f'{cfg.env.configuration_name}_latest_buffer')
    else:  # periodic checkpoint
        checkpoint_path = os.path.join(ckpt_dir, f'{cfg.env.configuration_name}_frame_{collected_frames}.pt')
        buffer_dir = os.path.join(ckpt_dir, f'{cfg.env.configuration_name}_frame_{collected_frames}_buffer')
    
    # Main checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
        'collected_frames': collected_frames,
        'update_counter': update_counter,
        'highest_reward_tracking': highest_reward_tracking.tolist() if hasattr(highest_reward_tracking, 'tolist') else highest_reward_tracking,
        'exploration_policy_state': exploration_policy[1].state_dict() if hasattr(exploration_policy[1], 'state_dict') else None,
    }
    
    # Save main checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save replay buffer using TorchRL's native dumps() method
    if cfg.env.agent not in ['PPO']:
        try:
            buffer_length = len(replay_buffer)
            if buffer_length > 0:
                print(f"Saving replay buffer with {buffer_length} experiences using TorchRL dumps()...")
                
                # Remove existing buffer directory to avoid conflicts
                if os.path.exists(buffer_dir):
                    shutil.rmtree(buffer_dir)
                
                # Use TorchRL's native dumps method - this handles everything properly
                replay_buffer.dumps(buffer_dir)
                
                print(f"✓ Checkpoint saved: {checkpoint_path}")
                print(f"✓ Buffer saved: {buffer_dir} ({buffer_length} experiences)")
                
            else:
                print(f"✓ Checkpoint saved: {checkpoint_path}")
                print("ℹ Buffer is empty, no buffer directory created")
                
        except Exception as e:
            print(f"⚠ Warning: Could not save replay buffer: {e}")
            print(f"✓ Checkpoint saved (without buffer): {checkpoint_path}")
    else:
        print('No replay buffer saved for PPO agents (on-policy method)!')


def load_checkpoint(cfg, model, optimizer_actor, optimizer_critic, exploration_policy, device):
    """
    Load training checkpoint using TorchRL's native loads() method
    """
    ckpt_dir = os.path.join(cfg.env.data_path, 'checkpoints')
    checkpoint_path = os.path.join(ckpt_dir, f'{cfg.env.configuration_name}_latest.pt')
    buffer_dir = os.path.join(ckpt_dir, f'{cfg.env.configuration_name}_latest_buffer')
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None, None, None, None
    
    try:
        # Load main checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer states
        optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        
        # Load exploration policy state if available
        if checkpoint.get('exploration_policy_state') is not None and hasattr(exploration_policy[1], 'load_state_dict'):
            exploration_policy[1].load_state_dict(checkpoint['exploration_policy_state'])
        
        # Convert highest_reward_tracking back to numpy array if needed
        import numpy as np
        highest_reward_tracking = np.array(checkpoint['highest_reward_tracking'])
        
        # Load replay buffer using TorchRL's native loads() method
        replay_buffer = None
        if os.path.exists(buffer_dir) and cfg.env.agent not in ['PPO']:
            try:
                print(f"Loading replay buffer from {buffer_dir}...")
                
                # Create a fresh replay buffer with the same configuration
                replay_buffer = make_replay_buffer(
                    batch_size=cfg.optim.batch_size,  # Use the batch size from config
                    prb=cfg.replay_buffer.prb,
                    buffer_size=cfg.replay_buffer.size,
                    device="cpu",
                    agent=cfg.env.agent
                )
                
                # Use TorchRL's native loads method to restore the buffer
                replay_buffer.loads(buffer_dir)
                
                restored_length = len(replay_buffer)
                print(f"✓ Successfully restored {restored_length} experiences to replay buffer")
                
            except Exception as e:
                print(f"⚠ Warning: Could not load replay buffer: {e}")
                print("Creating fresh empty buffer...")
                
                # Create fresh buffer if loading fails
                replay_buffer = make_replay_buffer(
                    batch_size=cfg.optim.batch_size,
                    prb=cfg.replay_buffer.prb,
                    buffer_size=cfg.replay_buffer.size,
                    device="cpu",
                    agent=cfg.env.agent
                )
        else:
            print(f"⚠ No replay buffer found at {buffer_dir}")
            print("Creating fresh empty buffer...")
            
            # Create fresh buffer
            replay_buffer = make_replay_buffer(
                batch_size=cfg.optim.batch_size,
                prb=cfg.replay_buffer.prb,
                buffer_size=cfg.replay_buffer.size,
                device="cpu",
                agent=cfg.env.agent
            )
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        
        return (checkpoint['collected_frames'], 
                checkpoint['update_counter'], 
                highest_reward_tracking, 
                replay_buffer)
                
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return None, None, None, None