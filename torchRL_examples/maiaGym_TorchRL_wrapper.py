import os 
import sys

# Navigate up to the parent directory of 'maiaGym'
maiaGym_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(maiaGym_parent_dir)
sys.path.insert(0, maiaGym_parent_dir)

from typing import Optional
import maiaGym

from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from collections import defaultdict

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
    TensorDictPrimer,
    DoubleToFloat,
    Compose,
    ParallelEnv
)

from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
import torch

import numpy as np

from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, ObservationNorm
    
# ------------------------------------------------------------------------------------------------------------------------------
### TORCH-RL wrapper
# ------------------------------------------------------------------------------------------------------------------------------
def _step(self, tensordict):

    if tensordict['terminated']:
        obs, reward, done = self.base_env.terminate()
        img_data = np.ones((1,))

    else:
        action = tensordict["action"].detach().cpu().numpy().tolist()
        obs, reward, done, truncated, _ = self.base_env.step(action)
        # if self.render:
        #     obs, reward, done, img_data = self.base_env.step(tensordict["action"].detach().cpu().numpy())   
        # else:
        #     obs, reward, done = self.base_env.step(tensordict["action"].detach().cpu().numpy())
        #     img_data = np.ones((1,))
        # print('reward - ', reward, 'observation - ', obs, flush=True)

    out = TensorDict(
        {
            "observation": torch.tensor(obs).float(),
            "params": tensordict["params"],
            "reward": torch.tensor(reward).float(),
            "done": bool(done),
            "info": {},
            # "img": torch.tensor(img_data).float()
        },
        tensordict.shape,
    )

    return out

def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty() or "params" not in tensordict.keys():
        tensordict = self.gen_params(batch_size=self.batch_size)

    obs, _ = self.base_env.reset()
    # if self.render:    
    #     obs, img_data = self.base_env.reset()
    # else:
    #     obs = self.base_env.reset()
    #     img_data = np.ones((1,))

    out = TensorDict(
        {
            "observation": torch.tensor(obs).float(),
            # "img": torch.tensor(img_data).float(),
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out

def _make_spec(self, td_params, action_space, observation_space):
    self.observation_spec = CompositeSpec(
        observation=BoundedTensorSpec(
            low=-torch.inf,
            high=torch.inf,
            shape=(observation_space.shape),
            dtype=torch.float32,
        ),
        # img=BoundedTensorSpec(
        #     low=-torch.inf,
        #     high=torch.inf,
        #     shape=(), #(1,1)
        #     dtype=torch.float32,
        # ),
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )

    self.state_spec = self.observation_spec.clone()

    self.action_spec = BoundedTensorSpec(
        low=action_space.low,
        high=action_space.high,
        shape=action_space.shape,
        dtype=torch.float32,
    )

    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape,1))

def make_composite_from_td(td):
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng

def gen_params(batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as time steps."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "dt": 0.1,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class MAIA_FlowEnv_TorchRL(EnvBase):
    metadata = {
        "render_fps": 30,
    }
    batch_locked = True

    def __init__(self, environment, td_params=None, seed=None, env_config={}, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        self.render = env_config['render']
        
        envs = {
            'cylinder': maiaGym.Cylinder,
            'rotary_cylinder': maiaGym.RotaryCylinder,
            'pinball': maiaGym.Pinball,
            'jet_pinball': maiaGym.JetPinball,
            'naca0012': maiaGym.NACA0012,
            'cavity': maiaGym.Cavity,
            'cavity3Jet': maiaGym.Cavity3Jet,
            # 'step': hydrogym.firedrake.Step,
                }

        self.base_env = envs[environment](env_config=env_config)

        super().__init__(device=device, batch_size=[])

        self._make_spec(td_params, self.base_env.action_space, self.base_env.observation_space)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.num_inputs = self.base_env.num_inputs
    
    def finish_run(self):
        self.base_env.maiaInterface.finishRun()

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    # _step = staticmethod(_step)
    _step = _step
    _set_seed = _set_seed

# ------------------------------------------------------------------------------------------------------------------------------
### helper functions TORCH-RL wrapper
# ------------------------------------------------------------------------------------------------------------------------------

def make_MAIA_FlowEnv_torchrl(environment, env_config={}):
    
    env = MAIA_FlowEnv_TorchRL(environment=environment, 
                               env_config=env_config)

    return env

def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            # RewardSum(),
        ),
    )
    return transformed_env

# ------------------------------------------------------------------------------------------------------------------------------

def main():

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt 

    import omegaconf
    import argparse
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Property File", required=True)
    parser.add_argument("--numTimesteps", help="Property File", default=100)

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

    train_env = make_MAIA_FlowEnv_torchrl(environment=cfg.maia.environment,
                                          env_config=env_config)

    # train_env = TransformedEnv(train_env, ObservationNorm(in_keys=["observation"]))
    # train_env.transform.init_stats(cfg.env.max_episode_steps_validation * 5)
    train_env.set_seed(cfg.env.seed)
    train_env = apply_env_transforms(train_env, max_episode_steps=cfg.env.max_episode_steps)

    print('--- Testing single stepping in TorchRL wrapper')
    rewards, observations = [], []
    _data_eval = train_env.reset()
    for i in tqdm.tqdm(range(args.numTimesteps)):
        action = torch.zeros(train_env.num_inputs) 
        _data_eval['action'] = action
        _data_eval = train_env.step(_data_eval)
        rewards.append(_data_eval['next','reward'].numpy())
        _data_eval = step_mdp(_data_eval, keep_other=True)
        observations.append(_data_eval['observation'].numpy())
    
    rewards = np.stack(rewards)
    eval_reward = rewards.mean()
    observations = np.stack(observations)
    print('Single stepping: observation shape:', observations.shape, '| reward shape:', rewards.shape, '| mean reward:', eval_reward)
    print('Single stepping: max observation:', np.max(observations, axis=0), '| min observation:', np.min(observations, axis=0))

    print('--- Testing environment rollout in TorchRL wrapper')
    train_env.reset()
    rollout = train_env.rollout(args.numTimesteps)
    observations = rollout['observation'].numpy()
    
    print('Rollout: observation shape:', observations.shape)
    print('Rollout: max observation:', np.max(observations, axis=0), '| min observation:', np.min(observations, axis=0))
    print('--------------------- ROLLOUT ---------------------')
    print(rollout)
    
    train_env.finish_run()

if __name__ == "__main__":
    main()