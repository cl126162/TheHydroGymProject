import warnings
import sys
import os

# Navigate up to the parent directory of 'maiaGym'
maiaGym_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, maiaGym_parent_dir)

from maiaGym.maia_env_core import MaiaFlowEnv
import numpy as np
from einops import rearrange
import gym

class CavityBase(MaiaFlowEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        
        self.reward_loc=env_config.get('reward_loc')
        self.reward_scale=env_config.get('reward_scale')            
        
        # set default values if necessary
        self.generate_general_defaults()
    
    def get_reward(self):
        reward = np.sum(((self.obs - np.array(self.reward_loc)) / self.reward_scale)** 2)
        obj_dict = {}
        return -reward, obj_dict
    
    def generate_general_defaults(self):
        """Set default values based on Re and nDim"""
        
        # Re number for default environment configuration, if simulated Re number 
        # differs from default values in dict AND no other environment parameters are given
        self.Re_default = 4200
        
        self.default_values = {
            (4200, 2): {  # Defaults for Re=4140, nDim=2
                'num_substeps_per_iteration': 400,
                'xMin': 26.0,
                'xMax': 42.0,
                'nXProbes': 5,
                'yMin': -0.625,
                'yMax': 0.625,
                'nYProbes': 5,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 100,
            },
            (7500, 2): {  # Defaults for Re=1000, nDim=2
                'num_substeps_per_iteration': 200,
                'xMin': 26.0,
                'xMax': 42.0,
                'nXProbes': 5,
                'yMin': -0.625,
                'yMax': 0.625,
                'nYProbes': 5,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 100,
            },
        }

class Cavity(CavityBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.numJetsInSimulation = self._getProperty(self.runTime_propertyFileData, "lbNoJets")
        
        # Cylinder default values
        self.environment_default_values = {
            (4200, 2): {  # Defaults for Re=200, nDim=2
                'num_inputs': 1,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([0.0, 0.0]),
                'obs_scale': np.array([0.0, 0.0]),
            },
            (7500, 2): {  # Defaults for Re=7500, nDim=2
                'num_inputs': 1,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([0.0, 0.0]),
                'obs_scale': np.array([0.0, 0.0]),
            }
        }
        
        # merge general default values with environment default values
        self.merge_defaults(self.environment_default_values)
        
        # set default values if necessary
        self.set_defaults()
        
        # configure observation and action space
        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()
        
        self.set_default_normalization_factors()
        
        self.set_reward_normalization_factors()
        
        # self.reset()
    
    def convert_action(self, action):        
        return action
    
    def set_reward_normalization_factors(self):
        if self.reward_scale is None or self.reward_loc is None or (len(self.reward_loc) != self.num_outputs) or (len(self.reward_loc) != self.num_outputs):    
            print('WARNING: No correct normalization values provided. Trying to fetch default values!')
            # Fetch default dict using Re and nDim
            default_config = self.default_values.get((self.Re, self.nDim))
            if default_config is None:
                default_config = self.default_values.get((self.Re_default, self.nDim))
            
            # compute loc and scale for observation normalization
            if (self.observation_type == default_config.get('observation_type')) and (self.probe_locations == self.default_probe_locations):
                if self.Re in (key[0] for key in self.default_values.keys()):
                    print(f'Pre-calculated reward loc and scale found in default library for Re={self.Re}, observation_type={self.observation_type} and given probe locations!', flush=True)  
                    self.reward_loc = self._set_default(self.reward_loc, default_config.get('reward_loc'), 'reward_loc', force_overwrite=True)
                    self.reward_scale = self._set_default(self.reward_scale, default_config.get('reward_scale'), 'reward_scale', force_overwrite=True)
            else:
                print(f'No reward loc and scale values found in default library for Re={self.Re}, observation_type={self.observation_type}, and given probe locations!', flush=True)
                print(f'Starting calculation of reward normalization factor for Re={self.Re} and observation_type={self.observation_type}!', flush=True)
                
                if self.nDim == 3:
                    print('WARNING: Calculation of normalization factor may take long. Consider pre-calculation!', self.probe_locations, flush=True)
                
                # compute normalization factors
                self.compute_normalization_factors(zero_actuation=True)        
                print('Computed reward loc values:', self.reward_loc.tolist(), flush=True)
                print('Computed reward scale values:', self.reward_scale.tolist(), flush=True)
        else:
            print('Using given loc and scale values for reward normalization', flush=True)

class Cavity3Jet(CavityBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.numJetsInSimulation = self._getProperty(self.runTime_propertyFileData, "lbNoJets")
        
        # Cylinder default values
        self.environment_default_values = {
            (4140, 2): {  # Defaults for Re=200, nDim=2
                'num_inputs': 3,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([0.0, 0.0]),
                'obs_scale': np.array([0.0, 0.0]),
            },
            (7500, 2): {  # Defaults for Re=7500, nDim=2
                'num_inputs': 3,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([0.0, 0.0]),
                'obs_scale': np.array([0.0, 0.0]),
            }
        }
        
        # merge general default values with environment default values
        self.merge_defaults(self.environment_default_values)
        
        # set default values if necessary
        self.set_defaults()
        
        # configure observation and action space
        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()
        
        self.set_default_normalization_factors()
        
        self.set_reward_normalization_factors()
    
    def convert_action(self, action):        
        return action
    
    def set_reward_normalization_factors(self):
        if self.reward_scale is None or self.reward_loc is None or (len(self.reward_loc) != self.num_outputs) or (len(self.reward_loc) != self.num_outputs):    
            print('WARNING: No correct normalization values provided. Trying to fetch default values!')
            # Fetch default dict using Re and nDim
            default_config = self.default_values.get((self.Re, self.nDim))
            if default_config is None:
                default_config = self.default_values.get((self.Re_default, self.nDim))
            
            # compute loc and scale for observation normalization
            if (self.observation_type == default_config.get('observation_type')) and (self.probe_locations == self.default_probe_locations):
                if self.Re in (key[0] for key in self.default_values.keys()):
                    print(f'Pre-calculated reward loc and scale found in default library for Re={self.Re}, observation_type={self.observation_type} and given probe locations!', flush=True)  
                    self.reward_loc = self._set_default(self.reward_loc, default_config.get('reward_loc'), 'reward_loc', force_overwrite=True)
                    self.reward_scale = self._set_default(self.reward_scale, default_config.get('reward_scale'), 'reward_scale', force_overwrite=True)
            else:
                print(f'No reward loc and scale values found in default library for Re={self.Re}, observation_type={self.observation_type}, and given probe locations!', flush=True)
                print(f'Starting calculation of reward normalization factor for Re={self.Re} and observation_type={self.observation_type}!', flush=True)
                
                if self.nDim == 3:
                    print('WARNING: Calculation of normalization factor may take long. Consider pre-calculation!', self.probe_locations, flush=True)
                
                # compute normalization factors
                self.compute_normalization_factors(zero_actuation=True)        
                print('Computed reward loc values:', self.reward_loc.tolist(), flush=True)
                print('Computed reward scale values:', self.reward_scale.tolist(), flush=True)
        else:
            print('Using given loc and scale values for reward normalization', flush=True)
                
            