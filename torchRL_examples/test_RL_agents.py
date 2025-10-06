import warnings

import os

import numpy as np
import torch
import torch.cuda
from tqdm import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from torchrl.envs.transforms import ObservationNorm

from utils_ppo import (
    log_metrics,
    make_collector,
    make_loss_module,
    make_replay_buffer,
    make_ppo_agent,
    make_ppo_optimizer,
)

from maiaGym_TorchRL_wrapper import make_MAIA_FlowEnv_torchrl, apply_env_transforms

from torchrl.envs import TransformedEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import omegaconf

def main():  # noqa: F821
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", type=str, required=True)
        parser.add_argument("--ckpt_path", type=str, required=True)
        parser.add_argument("--eval_rollout_steps", type=int, required=True)
        args = parser.parse_args()

        from omegaconf import OmegaConf

        print('--------------------------------------------------------------------------------------------', flush=True)
        print("Simulated environment and args:", args, flush=True)
        print('--------------------------------------------------------------------------------------------', flush=True)

        zeroTraining = False
        constantTraining = False
        
        cfg = OmegaConf.load(args.config_file)

        plot_directory = cfg.env.data_path + '/testing/'
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        device = torch.device(cfg.network.device)

        torch.manual_seed(cfg.env.seed)
        np.random.seed(cfg.env.seed)

        print('--------------------------------------------------------------------------------------------', flush=True)
        print('Loaded config file path:', args.config_file, flush=True)
        print('--------------------------------------------------------------------------------------------', flush=True)

        # # Velocity probes
        # xp = np.linspace(1.0, 8.0, 8)
        # yp = np.linspace(-0.75, 0.75, 5)
        # X, Y = np.meshgrid(xp, yp)
        # probe_list = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]
        # probe_locations = [item for sublist in probe_list for item in sublist]

        env_config = {
            "configuration_file": args.config_file,
            "is_testing": False,
            "render": cfg.maia.render,
            # set customized probes values
            # "probe_locations": probe_locations,
            # if applicable, set normalization values via obs_loc and obs_scale - otherwise will be computed
            # "obs_loc": obs_loc,
            # "obs_scale": obs_scale,
            }

        eval_env = make_MAIA_FlowEnv_torchrl(environment=cfg.maia.environment,
                                            env_config=env_config)
        eval_env.set_seed(cfg.env.seed)
        eval_env = apply_env_transforms(eval_env, max_episode_steps=cfg.env.max_episode_steps)
        
        # Create agent
        if cfg.env.agent == "PPO":
            from utils_ppo import make_ppo_agent
            model, exploration_policy = make_ppo_agent(cfg, eval_env, eval_env, device)
        elif cfg.env.agent == "SAC":
            from utils import make_sac_agent
            model, exploration_policy = make_sac_agent(cfg, eval_env, eval_env, device)
        elif cfg.env.agent == "A2C":
            from utils_a2c import make_a2c_agent
            model, exploration_policy = make_a2c_agent(cfg, eval_env, eval_env, device)
        elif cfg.env.agent == "DDPG":
            from utils_ddpg import make_ddpg_agent
            model, exploration_policy = make_ddpg_agent(cfg, eval_env, eval_env, device)
        elif cfg.env.agent == "TD3":
            from utils_td3 import make_td3_agent
            model, exploration_policy = make_td3_agent(cfg, eval_env, eval_env, device)
        else:
            raise NotImplementedError 

        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)   
        model = model.to(device)
        print('model loaded', flush=True)

        observations = []
        rewards = []
        flow_data = []
        action_log = []

        _data_eval = eval_env.reset()
        observations.append(_data_eval['observation'])
        print('environment resetted', flush=True)

        vis_output_path = cfg.env.data_path + "testing/npz_data/"

        if not os.path.exists(vis_output_path):
            os.makedirs(vis_output_path)
        else:
            existing_files = os.listdir(vis_output_path)
            for file in existing_files:
                os.remove(vis_output_path + file)

        if not zeroTraining and not constantTraining:
            print('-----------------------------------------------------------------------------')
            print('Starting policy rollout for ', args.eval_rollout_steps, 'steps.', flush=True)
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                eval_rollout = eval_env.rollout(
                    max_steps=args.eval_rollout_steps,
                    policy=model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                observations = eval_rollout["next", "observation"].numpy()
                rewards = eval_rollout["next", "reward"].numpy()
                action_log = eval_rollout["action"].numpy()

        if zeroTraining:
            print('--- uncontrolled rollout ---', flush=True)
            _data_eval = eval_env.reset()
            for i in tqdm(range(args.eval_rollout_steps + 1)):
            # for i in tqdm(range(49)):
                if cfg.hydrogym.environment == 'pinball':
                    action = torch.zeros(3)
                else:
                    action = torch.zeros(1) #torch.randn(1)
                # action = torch.ones(1) * actions[i]
                _data_eval['action'] = action
                _data_eval = eval_env.step(_data_eval)
                rewards.append(_data_eval['next','reward'].numpy())

                _data_eval = step_mdp(_data_eval, keep_other=True)
                observations.append(_data_eval['observation'].numpy())
                action_log.append(action)
            
            rewards = np.stack(rewards)
            eval_reward = rewards.mean()
            observations = np.stack(observations)
            # flow_data2 = np.stack(flow_data)
            action_log = np.stack(action_log)

            print('max observations:', np.max(observations, axis=0), 'min observations:', np.min(observations, axis=0), flush=True)
        
        action = torch.zeros(3)
        
        if constantTraining:
            for i in tqdm(range(args.eval_rollout_steps + 1)):
            # for i in tqdm(range(99)):    
                action = torch.ones(1) * 1.0 #torch.randn(1)
                # action = torch.ones(1) * actions[i]
                _data_eval['action'] = action
                _data_eval = eval_env.step(_data_eval)
                _data_eval = step_mdp(_data_eval, keep_other=True)
        
        eval_env.finish_run()

if __name__ == "__main__":
   main()