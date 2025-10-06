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

class NACA0012Base(MaiaFlowEnv):
    def __init__(self, env_config):
        super().__init__(env_config)            
        
        self.unperturbed_avg_drag = self.cfg.maia.unperturbed_avg_drag
        self.unperturbed_avg_lift = self.cfg.maia.unperturbed_avg_lift
        # set default values if necessary
        self.generate_general_defaults()
    
    def get_reward(self):
        # TODO: get forces of multiple objects
        forces = self.maiaInterface.getForce(self.bcId[0])

        # compute lift and drag coefficients
        nonDim_forces = self.compute_nondim_coefficients(forces=forces,
                                                                density=1.0, 
                                                                referenceVelocity=self.Ma/np.sqrt(3), 
                                                                projectionLength=self.referenceLength/self.dX)

        obj_dict = {
            'forces': forces,
        }

        return -np.abs(nonDim_forces[1] - self.unperturbed_avg_lift) - self.omega * np.abs(nonDim_forces[0] - self.unperturbed_avg_drag), obj_dict
    
    def generate_general_defaults(self):
        """Set default values based on Re and nDim"""
        
        # Re number for default environment configuration, if simulated Re number 
        # differs from default values in dict AND no other environment parameters are given
        self.Re_default = 100
        
        self.default_values = {
            (100, 2): {  # Defaults for Re=100, nDim=2
                'num_substeps_per_iteration': 400,
                'xMin': -2.25,
                'xMax': -0.25,
                'nXProbes': 5,
                'yMin': -2.25,
                'yMax': 3.75,
                'nYProbes': 7,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 25,
            },
            (1000, 2): {  # Defaults for Re=1000, nDim=2
                'num_substeps_per_iteration': 315,
                'xMin': 1.0,
                'xMax': 7.0,
                'nXProbes': 13,
                'yMin': -0.75,
                'yMax': 0.75,
                'nYProbes': 5,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 25,
            },
            (4000, 2): {  # Defaults for Re=4000, nDim=2
                'num_substeps_per_iteration': 3,
                'xMin': 1.0,
                'xMax': 8.0,
                'nXProbes': 4,
                'yMin': -0.75,
                'yMax': 0.75,
                'nYProbes': 3,
                'zMin': -0.75,
                'zMax': 0.75,
                'nZProbes': 3,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 25,
            },
            (100, 3): {  # Defaults for Re=100, nDim=3
                'num_substeps_per_iteration': 1,
                'xMin': -2.25,
                'xMax': -0.25,
                'nXProbes': 5,
                'yMin': -2.25,
                'yMax': 3.75,
                'nYProbes': 7,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 25,
            },
            (1000, 3): {  # Defaults for Re=1000, nDim=3
                'num_substeps_per_iteration': 640,
                'xMin': 1.0,
                'xMax': 8.0,
                'nXProbes': 4,
                'yMin': -0.75,
                'yMax': 0.75,
                'nYProbes': 3,
                'zMin': -0.75,
                'zMax': 0.75,
                'nZProbes': 3,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 25,
            },
            (4000, 3): {  # Defaults for Re=4000, nDim=3
                'num_substeps_per_iteration': 0,
                'xMin': 1.0,
                'xMax': 8.0,
                'nXProbes': 4,
                'yMin': -0.75,
                'yMax': 0.75,
                'nYProbes': 3,
                'zMin': -0.75,
                'zMax': 0.75,
                'nZProbes': 3,
                'observation_type': ['u', 'v'],
                'max_episode_steps': 25,
            }
        }

class NACA0012(NACA0012Base):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.numJetsInSimulation = self._getProperty(self.runTime_propertyFileData, "lbNoJets")
        
        # Cylinder default values
        self.environment_default_values = {
            (100, 2): {  # Defaults for Re=200, nDim=2
                'num_inputs': 3,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([0.05649585659977028, 0.05576791632371117, 0.05491048422243212, 0.054117540482468675, 0.05345017188193814, 0.05531871696797257, 0.054136248698200225, 0.05264044590040994, 0.05112057483010623, 0.04967468889792475, 0.05407779931835513, 0.05234427878087398, 0.049985596698680135, 0.04736149855378029, 0.04460093956414825, 0.05347284705548253, 0.05129040221034096, 0.04807591419417819, 0.044076998710156884, 0.03924906545816478, 0.05424485875048475, 0.05206273348837985, 0.0486166402277106, 0.043749135782283816, 0.036405685490684445, 0.05681839470340416, 0.05551279692262185, 0.0536065681736488, 0.05122649139737975, 0.048298134414684106, 0.060105378552910066, 0.05996050727003115, 0.0600640829619645, 0.06080029912287575, 0.06285087428581385, -0.003930752113602455, -0.00501673428839211, -0.006556453474938782, -0.008377889633921548, -0.010495308927918614, -0.0017342408671861998, -0.0027403777259054764, -0.004267128152977138, -0.006208369154302445, -0.008620129339729384, 0.0015482269457716856, 0.0008652890391157726, -0.00033627125473825937, -0.0021152404222030763, -0.0046633723478112015, 0.005512457620782797, 0.005496229562287253, 0.005190540850842571, 0.004299822996712485, 0.0023251771784397464, 0.010304793032490472, 0.011431620577877215, 0.013026662944130976, 0.014881265058652237, 0.016734828833425617, 0.014156603209668942, 0.016239779447100162, 0.019545309242043955, 0.0242893651150564, 0.031754981470327834, 0.015808084595171527, 0.01796344171994851, 0.02115598175925995, 0.025179635670574525, 0.03010747583293979]),
                'obs_scale': np.array([0.01617231091298797, 0.01590274787677439, 0.015587656419968515, 0.015292510786222959, 0.015030016228230962, 0.017314651271257742, 0.016885515201043824, 0.01635975827271775, 0.015838061920972778, 0.015343849683735478, 0.017796018363340885, 0.017198810581367754, 0.01643057427537841, 0.015620204557388355, 0.014798933588390914, 0.017721207616316882, 0.016997586072537846, 0.016017987277404347, 0.014907005351274474, 0.013670120203328652, 0.017220719027102694, 0.01646821376744297, 0.015402993493765452, 0.01410297799918662, 0.012491140046922302, 0.016387420265998, 0.01578851163320581, 0.01495938834629766, 0.013992178678139252, 0.01290189826389414, 0.015134851437485742, 0.014798358265041441, 0.014381982154415683, 0.013993530697262373, 0.013725725786648386, 0.005925135648481681, 0.006157762977466549, 0.006505751041840547, 0.006927578010962328, 0.007415579868863104, 0.004354057603189755, 0.004656915288369281, 0.005106873461424292, 0.0056556700717482885, 0.006298691055689496, 0.0027350890095102294, 0.003067236734540326, 0.003565313952118238, 0.004193261651457815, 0.004970406530244267, 0.0017895834275849002, 0.001987231045229836, 0.0023149517306085476, 0.0027960900082676316, 0.0035313447065137248, 0.0027181567413270867, 0.0027007551986306857, 0.0026971355519600764, 0.0027359791565556746, 0.002876752526700737, 0.004568262421870323, 0.00461273539949993, 0.004754646369316633, 0.005082647477102422, 0.005824911364021102, 0.006328683192654947, 0.006425269240653963, 0.006649125664858465, 0.007047567316504695, 0.0076947305289236274]),
            },
            (1000, 2): {  # Defaults for Re=1000, nDim=2
                'num_inputs': 3,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([]),
                'obs_scale': np.array([]),
            },
            (4000, 2): {  # Defaults for Re=4000, nDim=2
                'num_inputs': 3,
                'MAX_CONTROL': None,
                'obs_loc': None,
                'obs_scale': None,
            },
            (100, 3): {  # Defaults for Re=1000, nDim=3
                'num_inputs': 3,
                'MAX_CONTROL': 0.075,
                'obs_loc': np.array([]),
                'obs_scale': np.array([]),
            },
            (1000, 3): {  # Defaults for Re=1000, nDim=3
                'num_inputs': 3,
                'MAX_CONTROL': 0.125,
                'obs_loc': np.array([]),
                'obs_scale': np.array([]),
            },
            (4000, 3): {  # Defaults for Re=4000, nDim=3
                'num_inputs': 3,
                'MAX_CONTROL': None,
                'obs_loc': None,
                'obs_scale': None,
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
    
    def convert_action(self, action):        
        return action

                
            