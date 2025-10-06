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

class SphereBase(MaiaFlowEnv):
    def __init__(self, env_config):
        super().__init__(env_config)            
        
        # set default values if necessary
        self.generate_general_defaults()
    
    
    def get_reward(self):
        # Process all bcId entries
        rewards = []
        forces_list = []
        
        for bc_id in self.bcId:
            forces = self.maiaInterface.getForce(bc_id)
            nonDim_coefficients = self.compute_nondim_coefficients(
                forces=forces,
                density=1.0, 
                referenceVelocity=self.Ma/np.sqrt(3), 
                projectionLength=self.referenceLength/self.dX
            )
            
            reward = -np.abs(nonDim_coefficients[0]).sum() - self.omega * np.abs(nonDim_coefficients[1]).sum() - self.omega * np.abs(nonDim_coefficients[2]).sum()
            rewards.append(reward)
            forces_list.append(forces)
        
        # Single obj_dict with list of forces
        obj_dict = {'forces': forces_list}
        
        # Return single reward for single entry, list for multiple
        return (rewards[0] if len(self.bcId) == 1 else rewards), obj_dict
    
    def generate_general_defaults(self):
        """Set default values based on Re and nDim"""
        
        # Re number for default environment configuration, if simulated Re number 
        # differs from default values in dict AND no other environment parameters are given
        self.Re_default = 300
        
        self.default_values = {
            (300, 3): {  # Defaults for Re=200, nDim=3
                'num_substeps_per_iteration': 650,
                'xMin': 2.0,
                'xMax': 9.0,
                'nXProbes': 8,
                'yMin': -0.75,
                'yMax': 0.75,
                'nYProbes': 3,
                'zMin': -1.5,
                'zMax': 1.5,
                'nZProbes': 4,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 200,
            },
            (3700, 3): {  # Defaults for Re=3900, nDim=3
                'num_substeps_per_iteration': 450,
                'xMin': 2.0,
                'xMax': 9.0,
                'nXProbes': 8,
                'yMin': -0.75,
                'yMax': 0.75,
                'nYProbes': 3,
                'zMin': -1.5,
                'zMax': 1.5,
                'nZProbes': 4,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 200,
            }
        }

class Sphere(SphereBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.numJetsInSimulation = self._getProperty(self.runTime_propertyFileData, "lbNoJets")
        
        # Cube default values
        self.environment_default_values = {
            (300, 3): {  # Defaults for Re=200, nDim=3
                'num_inputs': 2,
                'MAX_CONTROL': 0.08,
                'obs_loc': np.array([0.0]),
                'obs_scale': np.array([0.0]),
            },
            (3700, 3): {  # Defaults for Re=200, nDim=3
                'num_inputs': 2,
                'MAX_CONTROL': 0.08,
                'obs_loc': np.array([0.0]),
                'obs_scale': np.array([1.0]),
            },
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
    
    def convert_action(self, action):
        return action