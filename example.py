import sys
import os

# Navigate up to the parent directory of 'maiaGym'
maiaGym_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, maiaGym_parent_dir)

import numpy as np
import omegaconf
import tqdm

import argparse
import maiaGym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Property File", required=True)
    parser.add_argument("--numTimesteps", help="Property File", default=100)
    parser.add_argument("--resetInterval", help="Property File", default=100)

    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_file)
    
    # --- probe locations ------------------------------------------------
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

    # set normalization values for observations - if necessary
    # default: no normalization = raw CFD results -> obs_loc=0.0, obs_scale=1.0
    # recommended: nomalization by U_inf -> obs_loc=cfg.maia.U_inf, obs_scale=1.0
    obs_loc = [0.0] * int(len(probe_locations) // getattr(cfg.maia, 'nDim', 2)) * len(cfg.maia.observation_type)
    obs_scale = [1.0] * int(len(probe_locations) // getattr(cfg.maia, 'nDim', 2)) * len(cfg.maia.observation_type)
    
    # set normalization values for reward - if necessary (only for cavity flow)
    # default: no normalization -> reward_loc=0.0, reward_scale=1.0
    reward_loc = [0.0] * int(len(probe_locations) // getattr(cfg.maia, 'nDim', 2)) * len(cfg.maia.observation_type)
    reward_scale = [1.0] * int(len(probe_locations) // getattr(cfg.maia, 'nDim', 2)) * len(cfg.maia.observation_type)
    
    env_config = {
        "configuration_file": args.config_file,
        "is_testing": False,
        "render": cfg.maia.render,
        # set customized probes values
        "probe_locations": probe_locations,
        "obs_loc": obs_loc,
        "obs_scale": obs_scale,
        "reward_loc": reward_loc,
        "reward_scale": reward_scale,
        }

    envs = {
            'cylinder': maiaGym.Cylinder,
            'rotary_cylinder': maiaGym.RotaryCylinder,
            'pinball': maiaGym.Pinball,
            'jet_pinball': maiaGym.JetPinball,
            'naca0012': maiaGym.NACA0012,
            'cavity': maiaGym.Cavity,
            'cavity3Jet': maiaGym.Cavity3Jet,
            'square_cylinder': maiaGym.SquareCylinder,
            'cube': maiaGym.Cube,
            'sphere': maiaGym.Sphere,
            }

    env = envs[cfg.maia.environment](env_config=env_config)
    
    # reset environment
    env.reset()
    
    for i in tqdm.tqdm(range(args.numTimesteps)):
        obs, reward, done, _, _ = env.step(action=np.random.uniform(-env.MAX_CONTROL, env.MAX_CONTROL, size=env.num_inputs))
        
        if done or i % args.resetInterval == 0:
            print('Reset environment',flush=True)
            env.reset()

    # close environment
    env.maiaInterface.finishRun()

if __name__ == "__main__":
   main()