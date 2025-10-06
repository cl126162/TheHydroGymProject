import sys
import os

# Navigate up to the parent directory of 'maiaGym'
maiaGym_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, maiaGym_parent_dir)

from maiaGym.maia_mpmd_interface import MaiaInterface
from mpi4py import MPI

import gym
import numpy as np

import shutil
import toml
import omegaconf
from einops import rearrange
import tqdm

class MaiaFlowEnv(gym.Env):
    def __init__(self, env_config):
        self.is_testing=env_config.get('is_testing', False)
        self.probe_locations=env_config.get('probe_locations')
        self.obs_loc=env_config.get('obs_loc')
        self.obs_scale=env_config.get('obs_scale')
        
        self.configuration_file = env_config.get('configuration_file')
        if self.configuration_file is None:
            raise ConfigError("Error: 'configuration_file' not found in environment configuration.")
        
        self.cfg = omegaconf.OmegaConf.load(self.configuration_file)        
        self.reset_ckpt_path=self.cfg.maia.restart
        self.runtime_property_file = self.cfg.maia.runtime_property_file
        
        self.num_substeps_per_iteration = self.cfg.maia.num_sim_substeps_per_actuation
        self.observation_type = self.cfg.maia.observation_type
        self.max_episode_steps = self.cfg.env.max_episode_steps
        self.num_inputs = self.cfg.maia.num_action_inputs * self.cfg.env.n_agents
        self.MAX_CONTROL = self.cfg.maia.max_control
        self.render = self.cfg.maia.render
        
        # load reward scaling
        try:
            self.omega = self.cfg.maia.omega
        except:
            self.omega = 0.0

        self.runTime_propertyFileData = self._readPropertyFile(self.runtime_property_file)
        self.Ma = self._getProperty(self.runTime_propertyFileData, "Ma")
        self.maxRfnmntLvl = self._getProperty(self.runTime_propertyFileData, "maxRfnmntLvl")
        self.reductionFactor = self._getProperty(self.runTime_propertyFileData, "reductionFactor")
        self.domainLength = self._getProperty(self.runTime_propertyFileData, "domainLength")
        self.referenceLength = self._getProperty(self.runTime_propertyFileData, "referenceLength")
        self.nDim = self._getProperty(self.runTime_propertyFileData, "nDim")
        self.dX = self.reductionFactor * self.domainLength / (2**self.maxRfnmntLvl)
        self.bcId = self._getProperty(self.runTime_propertyFileData, "lbBndCndIdx")
        self.Re = self._getProperty(self.runTime_propertyFileData, "Re")

        # init mpi communication
        self.comm_world = MPI.COMM_WORLD
        self.maiaInterface = MaiaInterface(self.nDim)
        self.maiaInterface.init_comm(self.comm_world)
        print('Python communicator initialized', flush=True)
        
        if self.Re != self.cfg.maia.Re:
            raise ConfigError(f"Error: Re numbers of configuration file (Re={self.cfg.maia.Re}) and property file (Re={self.Re}) do not match! Adjustment required!")
        
        # computing 5 episodes to compute normalization factors for observations 
        # Only if norm factors are not given or found in defaults!
        self.obs_norm_episodes = 5
    
    def configure_observations(self):
        pass
    
    def step(self, action=None):
        """Advance the state of the environment.  
        Args:
            action (Iterable[ActType], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ObsType, float, bool, dict]: obs, reward, done, info
        """
        action = [a * self.MAX_CONTROL for a in action]
        
        self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)
        self.maiaInterface.setControlProperties(self.convert_action(action=action))

        self.probeData = self.maiaInterface.getProbeData(self.probe_locations)
        self.probeData = rearrange(self.probeData, '(n p) -> n p', n=self.noProbes)
        
        self.obs = []
        if 'u' in self.observation_type:
            self.obs.append(self.probeData[:, 0])
        if 'v' in self.observation_type:
            self.obs.append(self.probeData[:, 1])
        if self.nDim ==2:
            if 'rho' in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if 'p' in self.observation_type:
                self.obs.append(self.probeData[:, 3])
        elif self.nDim ==3:
            if 'w' in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if 'rho' in self.observation_type:
                self.obs.append(self.probeData[:, 3])
            if 'p' in self.observation_type:
                self.obs.append(self.probeData[:, 4])
        else:
            print('WARNING: nDim =', self.nDim, '> 3. Something must be wrong!')
        if 'forces' in self.observation_type:
            forces = []
            for id in range(len(self.bcId)):
                forces.append(self.maiaInterface.getForce(self.bcId[id]))
            self.obs.append(np.stack(forces))
        
        self.obs = np.concatenate(self.obs)
        reward, obj_dict = self.get_reward()
        self.obs = (self.obs - self.obs_loc) / self.obs_scale

        self.iter += 1
        done = self.check_complete()
        info = {}

        self.maiaInterface.continueRun()

        return self.obs, reward, bool(done), bool(done), info
    
    def reset(self, seed=None, options={}):
        print('Resetting environment', flush=True)

        self.maiaInterface.runTimeSteps(1)
        self.maiaInterface.reinit()

        self.maiaInterface.setControlProperties(self.convert_action(action=np.zeros(shape=self.num_inputs)))
        self.probeData = self.maiaInterface.getProbeData(self.probe_locations)
        self.probeData = rearrange(self.probeData, '(n p) -> n p', n=self.noProbes)
        
        self.obs = []
        if 'u' in self.observation_type:
            self.obs.append(self.probeData[:, 0])
        if 'v' in self.observation_type:
            self.obs.append(self.probeData[:, 1])
        if self.nDim ==2:
            if 'rho' in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if 'p' in self.observation_type:
                self.obs.append(self.probeData[:, 3])
        elif self.nDim ==3:
            if 'w' in self.observation_type:
                self.obs.append(self.probeData[:, 2])
            if 'rho' in self.observation_type:
                self.obs.append(self.probeData[:, 3])
            if 'p' in self.observation_type:
                self.obs.append(self.probeData[:, 4])
        else:
            print('WARNING: nDim =', self.nDim, '> 3. Something must be wrong!')
        if 'forces' in self.observation_type:
            forces = []
            for id in range(len(self.bcId)):
                forces.append(self.maiaInterface.getForce(self.bcId[id]))
            self.obs.append(np.stack(forces))
            
        self.obs = np.concatenate(self.obs)
        self.obs = (self.obs - self.obs_loc) / self.obs_scale

        self.iter = 0
        info = {}
        
        self.maiaInterface.continueRun()

        return self.obs, info
    
    def convert_action(self, action):
        pass
    
    def get_reward(self):
        pass

    def check_complete(self):
        return self.iter > self.max_episode_steps
    
    def set_observation_action_spaces(self):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_outputs,),
            dtype=float,
        )
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_inputs,),
            dtype=float,
        )
    
    def configure_probe_dimensions(self):    
        # configure probe dimensions
        if self.nDim == 2:
            self.noProbeVars = 4 #u, v, rho, p
        elif self.nDim ==3:
            self.noProbeVars = 5 #u, v, w, rho, p
        self.noProbes = len(self.probe_locations) // self.nDim
        print('ENV: number of probes', self.noProbes)
        print('ENV: probe locations', self.probe_locations)
        
    def set_defaults(self):
        # Fetch default dict using Re and nDim
        default_config = self.default_values.get((self.Re, self.nDim))
        if default_config is None:
           print(f"WARNING: No default values for Re ({self.Re}) and nDim ({self.nDim}). Default values are taken from Re={self.Re_default} and nDim={self.nDim}.", 
                 flush=True)
           default_config = self.default_values.get((self.Re_default, self.nDim))
       
        #Generate probe locations
        if self.nDim == 2:
            self.default_probe_locations = self._generate_2d_probe_locations(
            default_config.get('xMin'),
            default_config.get('xMax'),
            default_config.get('yMin'),
            default_config.get('yMax'),
            default_config.get('nXProbes'),
            default_config.get('nYProbes')
            )
        elif self.nDim == 3:
            self.default_probe_locations = self._generate_3d_probe_locations(
            default_config.get('xMin'),
            default_config.get('xMax'),
            default_config.get('yMin'),
            default_config.get('yMax'),
            default_config.get('zMin'),
            default_config.get('zMax'),
            default_config.get('nXProbes'),
            default_config.get('nYProbes'),
            default_config.get('nZProbes'),
            )
        if self.probe_locations is None:
            self.probe_locations = self.default_probe_locations
            print('WARNING: No value provided for probe locations! Using default probe locations at:', self.probe_locations, flush=True)

        # Check for default values in the parameters
        self.num_substeps_per_iteration = self._set_default(self.num_substeps_per_iteration, default_config.get('num_substeps_per_iteration'), 'num_substeps_per_iteration')
        self.observation_type = self._set_default(self.observation_type, default_config.get('observation_type'), 'observation_type')
        self.max_episode_steps = self._set_default(self.max_episode_steps, default_config.get('max_episode_steps'), 'max_episode_steps')
        self.num_inputs = self._set_default(self.num_inputs, default_config.get('num_inputs'), 'num_inputs')
        self.MAX_CONTROL = self._set_default(self.MAX_CONTROL, default_config.get('MAX_CONTROL'), 'MAX_CONTROL')
        
    def merge_defaults(self, env_defaults):
        """Merges environment-specific default values into the general default values.

        This function iterates through the environment-specific default values and
        updates the corresponding entries in the general default values dictionary.
        If a key (Re, nDim) exists in both dictionaries, the environment-specific
        values will overwrite the general defaults for those keys. If a key exists
        only in the environment-specific defaults, it will be added to the general
        defaults.

        Args:
            env_defaults (dict): A dictionary of environment-specific default values,
                                 structured similarly to self.default_values.
        """
        for key, value in env_defaults.items():
            if key in self.default_values:
                # If the key exists in default_values, update the inner dictionary
                self.default_values[key].update(value)
            else:
                # If the key doesn't exist, add the entire entry
                self.default_values[key] = value
    
    def set_default_normalization_factors(self):
        if self.obs_scale is None or self.obs_loc is None or (len(self.obs_loc) != self.num_outputs) or (len(self.obs_loc) != self.num_outputs):    
            print('WARNING: No correct normalization values provided. Trying to fetch default values!')
            print('self.num_outputs', self.num_outputs)
            # Fetch default dict using Re and nDim
            default_config = self.default_values.get((self.Re, self.nDim))
            if default_config is None:
                default_config = self.default_values.get((self.Re_default, self.nDim))
            
            # compute loc and scale for observation normalization
            if (self.observation_type == default_config.get('observation_type')) and (self.probe_locations == self.default_probe_locations):
                if self.Re in (key[0] for key in self.default_values.keys()):
                    print(f'Pre-calculated loc and scale found in default library for Re={self.Re}, observation_type={self.observation_type} and given probe locations!', flush=True)  
                    self.obs_loc = self._set_default(self.obs_loc, default_config.get('obs_loc'), 'obs_loc', force_overwrite=True)
                    self.obs_scale = self._set_default(self.obs_scale, default_config.get('obs_scale'), 'obs_scale', force_overwrite=True)
            else:
                print(f'No loc and scale values found in default library for Re={self.Re}, observation_type={self.observation_type}, and given probe locations!', flush=True)
                print(f'Starting calculation of normalization factor for Re={self.Re} and observation_type={self.observation_type}!', flush=True)
                
                if self.nDim == 3:
                    print('WARNING: Calculation of normalization factor may take long. Consider pre-calculation!', self.probe_locations, flush=True)
                
                # compute normalization factors
                self.compute_normalization_factors()        
                print('Computed loc values:', self.obs_loc.tolist(), flush=True)
                print('Computed scale values:', self.obs_scale.tolist(), flush=True)
        else:
            print('Using given loc and scale values for observation normalization', flush=True)
    
    def compute_normalization_factors(self, zero_actuation=False):
        probes = np.zeros(shape=(self.obs_norm_episodes * self.max_episode_steps, 
                                 int(self.noProbes * self.noProbeVars)))
        
        for i in tqdm.tqdm(range(self.obs_norm_episodes * self.max_episode_steps)):
            if i % self.max_episode_steps == 0 and i > 0:
                # reset environment
                self.maiaInterface.runTimeSteps(1)
                self.maiaInterface.reinit()
                self.maiaInterface.setControlProperties(self.convert_action(action=np.zeros(shape=self.num_inputs)))                
                self.maiaInterface.continueRun()
                print('resetted environment')

            self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)
            if zero_actuation:
                self.maiaInterface.setControlProperties(self.convert_action(action=np.zeros(shape=self.num_inputs)))
            else:
                self.maiaInterface.setControlProperties(self.convert_action(action=np.random.uniform(-self.MAX_CONTROL, self.MAX_CONTROL, size=self.num_inputs)))

            probes[i, :] = self.maiaInterface.getProbeData(self.probe_locations)
            
            self.maiaInterface.continueRun()
        
        loc = np.mean(probes, axis=0)
        scale = np.std(probes, axis=0)
        
        loc = rearrange(loc, '(n p) -> n p', n=self.noProbes)
        scale = rearrange(scale, '(n p) -> n p', n=self.noProbes)
        
        self.obs_loc, self.obs_scale = [], []
        if 'u' in self.observation_type:
            self.obs_loc.append(loc[:, 0])
            self.obs_scale.append(scale[:, 0])
        if 'v' in self.observation_type:
            self.obs_loc.append(loc[:, 1])
            self.obs_scale.append(scale[:, 1])
        if self.nDim ==2:
            if 'rho' in self.observation_type:
                self.obs_loc.append(loc[:, 2])
                self.obs_scale.append(scale[:, 2])
            if 'p' in self.observation_type:
                self.obs_loc.append(loc[:, 3])
                self.obs_scale.append(scale[:, 3])
        elif self.nDim ==3:
            if 'w' in self.observation_type:
                self.obs_loc.append(loc[:, 2])
                self.obs_scale.append(scale[:, 2])
            if 'rho' in self.observation_type:
                self.obs_loc.append(loc[:, 3])
                self.obs_scale.append(scale[:, 3])
            if 'p' in self.observation_type:
                self.obs_loc.append(loc[:, 4])
                self.obs_scale.append(scale[:, 4])
        else:
            print('WARNING: nDim =', self.nDim, '> 3. Something must be wrong!')
        
        
        if zero_actuation:
            self.reward_loc = np.concatenate(self.obs_loc)
            self.reward_scale = np.concatenate(self.obs_scale)
        else:
            self.obs_loc = np.concatenate(self.obs_loc)
            self.obs_scale = np.concatenate(self.obs_scale)
    
    def configure_observations(self):
        self.num_outputs = 0
        self.num_probes = int(len(self.probe_locations) / self.nDim)

        if 'forces' in self.observation_type:
            self.num_outputs += self.nDim
        if 'u' in self.observation_type:
            self.num_outputs += self.num_probes
        if 'v' in self.observation_type:
            self.num_outputs += self.num_probes
        if 'w' in self.observation_type:
            self.num_outputs += self.num_probes
        if 'rho' in self.observation_type:
            self.num_outputs += self.num_probes
        if 'p' in self.observation_type:
            self.num_outputs += self.num_probes        
    
    def compute_nondim_coefficients(self, forces, 
                                    density:float=1.0, 
                                    referenceVelocity:float=0.1/np.sqrt(3), 
                                    projectionLength:float=20.0):
        """
        Compute non-dimensionalized force coefficients
        
        Args:
            [forces]: list of forces, e.g. [f_x, f_y]
            density: density value, default is 1.0
            referenceVelocity: reference velocity for non-dimenionalization scaled by sqrt(3)
                               due to LB solver
            projectionLength: reference length scale for non-dimenionalization
        
        return [force_coefficients]: list of non-dimensionalized force coefficients, e.g. [C_D, C_L]
        """

        force_coefficients = (2 * forces) / ( density * referenceVelocity**2 * projectionLength)
        return force_coefficients
        
    #---helper functions------------------------------------------------------------
    def _readPropertyFile(self, propertyFilePath:str):
        propertyFileData = None
        with open(propertyFilePath, 'r') as f:
            propertyFileData = toml.load(f)
        return propertyFileData

    def _getProperty(self, propertFileData, key:str):
        if isinstance(key, list):
           return propertFileData[key[0]][key[1]]
        else:
            return propertFileData[key]

    def _updateProperty(self, propertFileData, key:str, value):
        if isinstance(key, list):
            propertFileData[key[0]][key[1]] = value
        else:
            propertFileData[key] = value

    def _writePropertyFile(self, propertyFileData, propertyFilePath:str):
        with open(propertyFilePath, 'w') as f:
            toml.dump(propertyFileData, f)
    
    def copy_reset_ckpt(self, reset_ckpt_path):
        output_key = self._getProperty(self.runTime_propertyFileData, "outputDir")
        output_path = self._getProperty(output_key, "default") + 'restart_.Netcdf'
        shutil.copy2(reset_ckpt_path, output_path)
    
    def _set_default(self, value, default_value, variable_name, force_overwrite=False):
      """Set default if value is None"""
      if value is None or force_overwrite:
        print('WARNING: No value provided for', variable_name, '- value is set to:', default_value, flush=True)
        return default_value
      return value
  
    def _generate_2d_probe_locations(self, xMin, xMax, yMin, yMax, nXProbes, nYProbes):
        xp = np.linspace(xMin, xMax, nXProbes)
        yp = np.linspace(yMin, yMax, nYProbes)
        X, Y = np.meshgrid(xp, yp)
        probe_list = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]
        probe_locations = [item for sublist in probe_list for item in sublist]
        return probe_locations

    def _generate_3d_probe_locations(self, xMin, xMax, yMin, yMax, zMin, zMax, nXProbes, nYProbes, nZProbes):
        xp = np.linspace(xMin, xMax, nXProbes)
        yp = np.linspace(yMin, yMax, nYProbes)
        zp = np.linspace(zMin, zMax, nZProbes)
        X, Y, Z = np.meshgrid(xp, yp, zp)
        probe_list = [(x, y, z) for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())]
        probe_locations = [item for sublist in probe_list for item in sublist]
        return probe_locations

class ConfigError(Exception):
    pass


def main():
    import argparse
    import maiaGym

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Property File", required=True)
    parser.add_argument("--numTimesteps", help="Property File", default=100)
    parser.add_argument("--resetInterval", help="Property File", default=100)

    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_file)
    
    # --- probe locations ------------------------------------------------
    
    # --- Define Trapezoid Parameters ---
    x_left = 26.0
    y_min_left = -0.625
    y_max_left = 0.625

    x_right = 42.0
    y_min_right = -1.5
    y_max_right = 1.5

    # --- Define Probe Grid Size ---
    nXProbes = 5  # Number of probes horizontally
    nYProbes = 5  # Number of probes vertically *at each x*

    # --- Generate Probe Locations ---

    # 1. Generate the x-coordinates for the probes
    xp = np.linspace(x_left, x_right, nXProbes)
    probe_list = []

    # 3. Iterate through each x-coordinate
    for x in xp:
        # Calculate the interpolation factor (how far x is between x_left and x_right)
        if x_right != x_left:
            interp_factor = (x - x_left) / (x_right - x_left)
        else:
            interp_factor = 0

        # Linearly interpolate the y_min and y_max for the current x
        current_y_min = y_min_left + interp_factor * (y_min_right - y_min_left)
        current_y_max = y_max_left + interp_factor * (y_max_right - y_max_left)

        # 4. Generate the y-coordinates for the current x using the interpolated limits
        if nYProbes > 1:
            yp_at_x = np.linspace(current_y_min, current_y_max, nYProbes)
        elif nYProbes == 1:
            # If only one probe, place it in the middle of the y-range
            yp_at_x = np.array([(current_y_min + current_y_max) / 2.0])
        else: # nYProbes <= 0
            yp_at_x = np.array([]) # No probes in y direction

        for y in yp_at_x:
            probe_list.append((x, y))

    # xMin, xMax, nXProbes = -2.25, -0.25, 5
    # yMin, yMax, nYProbes = -2.25, 3.75, 7
    # xp = np.linspace(xMin, xMax, nXProbes)
    # yp = np.linspace(yMin, yMax, nYProbes)
    # X, Y = np.meshgrid(xp, yp)
    # probe_list = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]

    xMin, xMax, nXProbes = 43.0, 43.8, 3
    yMin, yMax, nYProbes = -5.0, -2.5, 3
    xp = np.linspace(xMin, xMax, nXProbes)
    yp = np.linspace(yMin, yMax, nYProbes)
    X, Y = np.meshgrid(xp, yp)
    probe_list = probe_list + [(x, y) for x, y in zip(X.ravel(), Y.ravel())]
    probe_locations = [item for sublist in probe_list for item in sublist]

    obs_loc = [0.0005641808198231202, 0.007714556860892183, 0.015626756524384756, 0.028102678843530297, 0.037107759448861716, 0.016956027879959135, 0.02072745027785437, 0.025858544250328717, 0.03422872929035132, 0.04231891543277754, 0.022381865903719194, 0.025930739704444407, 0.031414668333214116, 0.038648706275993774, 0.045637622546842414, 0.02002228071477175, 0.025239906623272714, 0.032149310327231284, 0.041003791582404775, 0.04766759864903593, 0.011790723876825443, 0.019625279230688956, 0.028475100493885297, 0.03855866481041434, 0.04579206011047922, -0.0008678346189032619, -0.0009244646187460345, -0.00022958740270003218, 8.946308462361838e-05, -0.0005832691452174871, -0.00025012664945875705, 0.0024345072426426207, 0.0008307636139794782, -0.00015429650086703014, 0.003030537159044286, 0.0016627954296234305, 0.0007269616943088781, -4.085429819279046e-05, -7.419077546785336e-05, 0.0045258658149204415, 0.003108692093740427, 0.002208149342669138, 0.0014867160810114817, 0.0012378958017995243, 0.0012545732951680269, 0.0009900361654442966, 0.0006245871073432875, 0.0003672878782604077, 0.00025860264597876037, -0.0023582204084126677, -0.001961185650628666, -0.0018325535181617856, -0.0017214976250546957, -0.0014925165607543587, -0.009592730009451506, -0.006772882777345725, -0.004337838513946083, -0.002184928836783536, -0.0011196868479610258, -0.027287745060846115, -0.027555913908148325, -0.014881543156981406, -0.025192168236435452, -0.027512138346365466, -0.017486941332640484, -0.02093683626706232, -0.02437304123031832, -0.019544370340617425]
    obs_scale = [0.008337788561987023, 0.011038553687361646, 0.0131984224964945, 0.014776883905090287, 0.014374552239966254, 0.009048777058156049, 0.009539863091586314, 0.009550899132782674, 0.009184845200696977, 0.009226867172536159, 0.007767155479083116, 0.008124673344700822, 0.007957444668343215, 0.007523211627478649, 0.007763600792204508, 0.008456812351496854, 0.008178130414170606, 0.008283593889715059, 0.008490738430382052, 0.007804236959878296, 0.008829659135094765, 0.0088415776156003, 0.008845819584726522, 0.00789333684327556, 0.006262079992951094, 0.005364604891614847, 0.003688640682412969, 0.0005904695334016081, 0.005519690033462697, 0.004189574658597623, 0.0008092668459798694, 0.0059009362200028595, 0.004424037568651984, 0.0010695748370776823, 0.01280798143814043, 0.014044362575677337, 0.014073789685795332, 0.012887059161013303, 0.01132356679352438, 0.01165001632031309, 0.01355488028473276, 0.01449966734678577, 0.014674623324377172, 0.013632457024360771, 0.010020471110743766, 0.012194269988458906, 0.013848430605544417, 0.014457381955104112, 0.013828811561689726, 0.010238221891709823, 0.01215309819886226, 0.013304974070626165, 0.013256071827714131, 0.012147462097803766, 0.00923325599438774, 0.01033682665456283, 0.0106100107204179, 0.010275006496619082, 0.00941135729920286, 0.008616827169173835, 0.008400182350980463, 0.005993800684235483, 0.009062119106846862, 0.010146042428759857, 0.008022434759535197, 0.00958467305293726, 0.011254464024241791, 0.01019096233096713]

    reward_loc = [-0.0029267366926722964, 0.0011065325925263907, 0.008959070420732122, 0.028923495027268993, 0.04406954168004567, 0.010599966293681401, 0.009466166357786047, 0.015104141590643575, 0.03254698636296816, 0.048639819397009024, 0.020807436424358696, 0.020420038004356857, 0.02466122911317522, 0.03643346489407995, 0.04970024503755621, 0.01873359798525915, 0.022836584845155654, 0.029904249862702275, 0.041506347193028204, 0.050896901133131923, 0.010829054240389397, 0.018750435654269492, 0.028398646193754495, 0.04047719829818425, 0.04909923069507615, -0.00014897553715092053, -0.0005438272142418032, -0.00016769321768312842, 0.00044459805014195747, -0.00032156235401322673, -0.00019436573150389023, 0.002102949673794072, 0.0006369806895282507, -0.00017480553240497248, 0.0004599854280040659, 0.0004905352562221559, 0.0001840158893905479, -0.00020550585047910067, -3.1270549051956736e-05, 0.004081522572190128, 0.0020972702298247717, 0.0008633973020255411, 0.00041141222779594205, 0.0007505245839255296, 0.002142487568831706, 0.0017016817621707105, 0.000778600688948078, 7.410893751030673e-05, 0.00020407703268690027, -0.0009805021277999055, -0.0007959473977072085, -0.001133250022029815, -0.001430401364532087, -0.001204476513711266, -0.007326835888901, -0.004643648252656018, -0.0026519413049277433, -0.0009055504774932269, -9.252448002590531e-05, -0.02422369554902625, -0.026308906160704717, -0.014354133922189614, -0.021721993108638423, -0.02520684828108789, -0.016287154543857787, -0.017859980004083672, -0.02193064564641898, -0.018278634989326113]
    reward_scale = [0.00043138387925547515, 0.0004938589825170338, 0.0014870109401939184, 0.0024987139022107853, 0.0019011131281135311, 0.0025832182440118146, 0.0018870992311196873, 0.003497765306478497, 0.005973373202383357, 0.0029074368866225226, 0.0032737680039532945, 0.0037901316240697143, 0.003298168768318105, 0.006822635375842784, 0.004064822756818092, 0.00470731425024174, 0.002840467166317693, 0.004470125163025056, 0.005486351628376063, 0.0036840260964062536, 0.0046972732532237745, 0.0036289677435447994, 0.0035495030608403387, 0.003497743759332421, 0.002861777962820763, 0.0017571720583438397, 0.0014996669264055963, 0.0003014204166851903, 0.0022040413615154507, 0.0017572784320157857, 0.0003538491484031893, 0.003539770221954411, 0.002717489729759218, 0.0005784881114128231, 0.000980736145452151, 0.0010781622459044504, 0.0010569581786153699, 0.0009442958824803161, 0.0009336336452798979, 0.0022185307913335835, 0.002805507389632472, 0.0028278715433509537, 0.0023724150252652327, 0.002154820687427701, 0.00253606923133631, 0.0036818862613419647, 0.0044903279913826874, 0.004283752756722287, 0.0037458975026533293, 0.004646017144882061, 0.0064391213030334, 0.0071485587954657525, 0.006308011714979438, 0.0050257475102315615, 0.005840364711405544, 0.006940653551810533, 0.006805339763956311, 0.0057276875155616944, 0.004512303437669484, 0.0024342022521963128, 0.0035473261159514775, 0.003011071123306238, 0.0015348930632677526, 0.0027784230698855105, 0.003968827015957057, 0.0032281980295872205, 0.0033758827646565154, 0.0028341397231501555]
    
    env_config = {
        "configuration_file": args.config_file,
        "is_testing": False,
        "render": cfg.maia.render,
        # set customized probes values
        "probe_locations": probe_locations,
        # if applicable, set obs normalization values via obs_loc and obs_scale - otherwise will be computed
        "obs_loc": obs_loc,
        "obs_scale": obs_scale,
        # if applicable, set reward normalization values via reward_loc and reward_scale - otherwise will be computed -> only for cavity flow case !!!
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
            }

    env = envs[cfg.maia.environment](env_config=env_config)
    
    rewards = []
    output = []
    
    env.reset()
    
    for i in tqdm.tqdm(range(args.numTimesteps)):
            # obs, reward, done, _, _ = env.step(action=np.zeros(shape=env.num_inputs))
            # obs, reward, done, _, _ = env.step(action=np.ones(shape=env.num_inputs)* 0.75)
            obs, reward, done, _, _ = env.step(action=np.random.uniform(-env.MAX_CONTROL, env.MAX_CONTROL, size=env.num_inputs))
            
            if done or i % args.resetInterval == 0:
                print('reset environment',flush=True)
                env.reset()
            rewards.append(reward)
            output.append(obs)
    
    # print(obs)
    rewards = np.stack(rewards)
    output = np.stack(output)

    env.maiaInterface.finishRun()
    
    # np.save('rewards_zeroActuation.npy', rewards)
    zeroRewards = np.load('rewards_zeroActuation.npy')
    
    # --- plotting
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt # Optional: for visualization
    
    timesteps = np.arange(len(rewards)) # Assumes both arrays have the same length

    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    plt.plot(timesteps, rewards, label='Rewards', color='blue', linestyle='-')
    plt.plot(timesteps, zeroRewards, label='Zero Rewards', color='red', linestyle='--')
    plt.title('Comparison of Reward Types Over 1 Episode') 
    plt.xlabel('Timestep Number')                     
    plt.ylabel('Reward Value')                      
    plt.legend()                                   
    plt.grid(True)
    plt.savefig('reward_comparison.png')

if __name__ == "__main__":
   main()
    