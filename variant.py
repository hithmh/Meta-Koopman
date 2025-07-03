import gym
import numpy as np
import replay_memory_meta_simulator
import replay_memory


SEED = None

VARIANT = {
    # Environment name
    # 'env_name': 'cartpole_cost',
    # 'env_name': 'oscillator',
    'env_name': 'three_tank',

    # training prams
    # 'alg_name': 'Koopman_v10',
    # 'alg_name': 'Koopman_v10_meta',
    'alg_name': 'Koopman_meta_deterministic_tanh',
    # 'alg_name': 'Koopman_meta_deterministic',

    #Description
    'additional_description': '',

    'train_model': True,
    'continue_training': False,
    'eval_control': False,

    # 'store_hyperparameter':True,
    'store_hyperparameter': False,  # store hyperparameters even while evaluation
    'save_frequency': 5,

    'import_saved_data': False,
    'save_data': True,

    'import_merging_saved_data': True,
    'continue_data_collection': False,
    'collect_data_with_controller': False,
    'evaluation_form': 'param_variation',
    'num_of_trials': 4,  # number of random seeds
    'eval_list': [
    ],
    'trials_for_eval': [str(i) for i in range(0, 1)],
}

VARIANT['log_path'] = '/'.join(['./log', VARIANT['env_name'], VARIANT['alg_name'] + VARIANT['additional_description']])

ENV_PARAMS = {
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        # 'eval_render': True,
        'eval_render': False,
        'num_of_task': 10,     # number of import tasks (for meta training only)

        ## DeSKO
        'reference': np.array([0, 0, 0, 0], dtype=np.float32),
        'Q': np.diag([0.001, .000, 0.5, 0.1]) * 1,
        'R': np.diag([.01]),

        ### MPC params
        # 'reference': np.array([0, 0, 0, 0], dtype=np.float32),
        # 'Q': np.diag([0.01, .001, 1., 0.1]) * 1,
        # 'R': np.diag([.01]),
        'end_weight': 10,
        'control_horizon': 25,
        'MPC_pred_horizon': 16,

        'apply_state_constraints': False,
        'apply_action_constraints': True,

        # 'lr': 1e-5,
        'lr_scaler': 0.005,
        'w_k_bound': 1e-6,
        'v_k_bound': 1e-4,
    },

    'three_tank': {
        'max_ep_steps': 500,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,
        'num_of_task': 20,  # number of import tasks (for meta training only)

        ### DeSKO params
        # 'reference': np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877],
        #                       dtype=np.float32),
        # 'Q': np.diag([1., 1., 0.000, 1., 1., 0.000, 1, 1., 0.000]),
        # 'R': np.diag(0.00001 * np.ones([3])) * 1,

        # ### MeKO params
        'reference': np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877], dtype=np.float32),
        'Q': np.diag([1., 1., 0.000, 1., 1., 0.000, 1, 1., 0.000]),
        'R': np.diag(0.001 * np.ones([3]))*1,
        'end_weight': 1.,
        'control_horizon': 35,
        'MPC_pred_horizon': 35,
        'apply_state_constraints': False,
        'apply_action_constraints': True,

        # 'lr_scaler': 0.8,
        # 'w_k_bound': 0,
        # 'v_k_bound': 0,

        'lr_scaler': 0.8,
        'w_k_bound': 1e-8,
        'v_k_bound': 1e-8,
    },

    'oscillator': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,
        'num_of_task': 10,     # number of import tasks (for meta training only)


        ### DeSKO-meta params
        'reference': np.array([0, 0, 0, 6, 0, 0], dtype=np.float32),
        'Q': np.diag([0., 0.,  0., 1., 0., 0.]),
        'R': np.diag(0.01*np.ones([3])),
        'end_weight': 10.,
        'control_horizon': 10,
        'MPC_pred_horizon': 16,
        # ### MPC params
        # 'reference': np.array([0, 0, 0, 6, 0, 0], dtype=np.float32),
        # 'Q': np.diag([0., 0.,  0., 1., 0., 0.]),
        # 'R': np.diag(0.01*np.ones([3])),
        # 'end_weight': 10.,
        # 'control_horizon': 10,
        # 'MPC_pred_horizon': 30,
        'apply_state_constraints': False,
        'apply_action_constraints': True,


        'lr_scaler': 0.9,
        'w_k_bound': 1e-3,
        'v_k_bound': 1e-3,
    },

}

ALG_PARAMS = {
    'Koopman_meta_deterministic': {
        'controller_name': 'Adaptive_MPC_v3',
        # 'controller_name': 'Adaptive_MPC', ### nominal adaptive MPC

        'iter_of_data_collection': 3,
        'learning_rate': 1e-4,
        'decay_rate': 0.95,
        'decay_steps': 5,

        'activation': 'relu',
        # 'encoder_struct': [512, 512],
        'encoder_struct': [128, 128],
        # 'encoder_struct': [256, 256],
        'latent_dim': 128,
        'pred_horizon': 16,
        'alpha': .1,

        'l2_regularizer': 1e-3,
        'val_frac': 0.1,
        'batch_size': 128,
        'num_epochs': 400,
        'total_data_size': 5e4,
        # 'total_data_size': 2000,
        'further_collect_data_size': 1000,

        'segment_of_test': 8,
        'n_subseq': 220,  # number of subsequences to divide each sequence into
        'store_last_n_paths': 10,  # number of trajectories for evaluation during training
        'start_of_trial': 0,
        'history_horizon': 0,
    },
    'Koopman_meta_deterministic_tanh': {
        'controller_name': 'Adaptive_MPC_tanh',

        'iter_of_data_collection': 3,
        'learning_rate': 1e-4,
        'decay_rate': 0.95,
        'decay_steps': 5,

        'activation': 'relu',
        'encoder_struct': [256, 256],
        'latent_dim': 128,
        'pred_horizon': 16,
        'alpha': .1,

        'l2_regularizer': 1e-3,
        'val_frac': 0.1,
        'batch_size': 128,
        'num_epochs': 400,
        'total_data_size': 1e5,
        'further_collect_data_size': 1000,

        'segment_of_test': 8,
        'n_subseq': 220,  # number of subsequences to divide each sequence into
        'store_last_n_paths': 10,  # number of trajectories for evaluation during training
        'start_of_trial': 0,
        'history_horizon': 0,
    },
    'Koopman_v10': {
        'controller_name': 'Robust_fast_dynamic_MPC',
        'estimator_name': 'Luenberger',
        # 'controller_name': 'Robust_fast_MPC',
        # 'controller_name': 'Stochastic_MPC_with_motion_planning',    #for HalfCheetah environment
        'n_of_random_seeds': 1,
        'iter_of_data_collection': 3,
        'learning_rate': 1e-4,
        'decay_rate': 0.9,
        'decay_steps': 10,
        'activation': 'relu',
        # 'activation': 'elu',
        'encoder_struct': [256, 256],
        # 'encoder_struct': [256, 128, 80],
        # 'encoder_struct': [256, 100, 80],
        'latent_dim': 20,
        'pred_horizon': 16,
        'alpha': .1,
        'target_entropy': -80.,
        'l2_regularizer': 0.01,
        'val_frac': 0.1,
        'batch_size': 128,
        'num_epochs': 400,       #400
        'total_data_size': 1e6, #40000
        'further_collect_data_size': 1000,
        'segment_of_test': 8,
        'n_subseq': 220,  # number of subsequences to divide each sequence into
        'store_last_n_paths': 10,  # number of trajectories for evaluation during training
        'start_of_trial': 0,
        'history_horizon': 0,
    },
}


EVAL_PARAMS = {
    ### param_variation is only applicable to the cartpole environment

    'param_variation': {

        'num_of_paths': 10,   # number of path for evaluation
    },

    'impulse': {
        'magnitude_range': np.arange(0.1, 0.5, .05),
        'num_of_paths': 1,   # number of path for evaluation
        'impulse_instant': 200,
    },

    'constant_impulse': {
        'magnitude_range': np.arange(80, 155, 5),       # cartpole
        'num_of_paths': 4,   # number of path for evaluation
        'impulse_instant': 20,
    },
    'distribution_eval': {
        'n_seeds': 1000,
    },


    'dynamic': {
        # 'eval_additional_description': '-R-new',
         'eval_additional_description': '-test',
        # 'eval_additional_description': '-setpoint-fixed',
        # 'eval_additional_description': '-L-test',
        # 'eval_additional_description': '-robustnesseval-smalltape-circle',
        # 'eval_additional_description': '-R-19.10.2021',
        'num_of_paths': 5,   # number of path for evaluation
        # 'plot_average': True,
        'plot_average': False,
        'directly_show': True,
        'dimension_of_interest': [6,7,8],

    },
}

for key in ENV_PARAMS[VARIANT['env_name']].keys():
    VARIANT[key] = ENV_PARAMS[VARIANT['env_name']][key]
for key in ALG_PARAMS[VARIANT['alg_name']].keys():
    VARIANT[key] = ALG_PARAMS[VARIANT['alg_name']][key]
for key in EVAL_PARAMS[VARIANT['evaluation_form']].keys():
    VARIANT[key] = EVAL_PARAMS[VARIANT['evaluation_form']][key]
# VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]



def get_env_from_name(args):
    name = args['env_name']
    if name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_random':
        from envs.ENV_V2 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_observation_noise':
        from envs.ENV_V3 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_partial':
        from envs.ENV_V4 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'turtle_v4':
        from envs.gym_pybullet_mobilerobot_v4 import MobileRoboGymEnv as env
        # from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = env(args['reference'])
        # env = gym.make('DeskoCartpole-v0', des_v=args['reference'])
        # env = dreamer()
        env = env.unwrapped
    elif name == 'turtle_v6':
        from envs.gym_pybullet_mobilerobot_v6 import MobileRoboGymEnv as env
        # from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = env(args['reference'])
        # env = gym.make('DeskoCartpole-v0', des_v=args['reference'])
        # env = dreamer()
        env = env.unwrapped
    elif name == 'trunk_arm':
        from envs.trunk_arm import TrunkArm as dreamer
        connect_to_real_system = False if args['import_saved_data'] else True
        env = dreamer(connect_to_real_system, args['reference'])
        env = env.unwrapped
    elif name == 'trunk_armV2':
        from envs.trunk_armV2 import TrunkArmV2 as dreamer
        connect_to_real_system = False if args['import_saved_data'] else True
        env = dreamer(connect_to_real_system, args['reference' ])
        env = env.unwrapped
    elif name == 'trunk_armV3':
        from envs.trunk_armV3 import TrunkArmV3 as dreamer
        connect_to_real_system = False if args['import_saved_data'] else True
        env = dreamer(connect_to_real_system, args['reference'])
        env = env.unwrapped
    elif name == 'trunk_armV4':
        from envs.trunk_armV4 import TrunkArmV4 as dreamer
        connect_to_real_system = False if args['import_saved_data'] else True
        env = dreamer(connect_to_real_system, args['reference'], args['MPC_pred_horizon'])
        env = env.unwrapped
    elif name == 'trunk_armV6':
        from envs.trunk_armV6 import TrunkArmV6 as dreamer
        connect_to_real_system = False if args['import_saved_data'] else True
        env = dreamer(connect_to_real_system, args['reference'], args['MPC_pred_horizon'])
        env = env.unwrapped
    elif name == 'trunk_arm_sim':
        from envs.trunk_arm_simulator import TrunkArmSimulator as dreamer
        if 'reference' in args.keys():
            env = dreamer(args['reference'])
        else:
            env = dreamer()
        env = env.unwrapped
    elif name == 'trunk_arm_sim_obs_noise':
        from envs.trunk_arm_simulatorv_2 import TrunkArmSimulator as dreamer
        if 'reference' in args.keys():
            env = dreamer(args['reference'])
        else:
            env = dreamer()
        env = env.unwrapped
    elif name == 'three_tank':
        from envs.three_tank import three_tank_system as dreamer
        env = dreamer(args['reference'])
        env = env.unwrapped
    elif name == 'three_tank_delay':
        from envs.three_tank_delay import three_tank_system as dreamer
        env = dreamer(args['reference'])
        env = env.unwrapped
    elif name == 'three_tank_partial':
        from envs.three_tank_partial import three_tank_system as dreamer
        env = dreamer(args['reference'])
        env = env.unwrapped
    elif name == 'three_tank_recycle_delay':
        from envs.three_tank_recycle_delay import three_tank_system as dreamer
        env = dreamer(args['reference'])
        env = env.unwrapped
    elif name == 'waste_water':
        from envs.waste_water_system import waste_water_system as dreamer
        env = dreamer(args['reference'])
        env = env.unwrapped
    elif name == 'linear_sys':
        from envs.linear_env import linear_env as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'Pendulum-v0':
        from envs.pendulum import PendulumEnv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'oscillator':
        from envs.oscillator import oscillator as env
        env = env(args['reference'])
        env = env.unwrapped

    elif name == 'oscillator_CME':
        from envs.oscillator_cme import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_no_input':
        from envs.oscillator_no_input import oscillator as env
        env = env(args['reference'])
        env = env.unwrapped
    elif name == 'MJS1':
        from envs.MJS1 import MJS as env
        env = env()
        env = env.unwrapped
    elif name == 'MJS2':
        from envs.MJS2 import MJS as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_complicated':
        from envs.oscillator_complicated import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_complicated_partial':
        from envs.oscillator_complicated import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_observation_noise':
        from envs.oscillator_observation_noise import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'oscillator_process_noise':
        from envs.oscillator_process_noise import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'minitaur':
        from envs.minitaur_env import minitaur_env as env
        env = env(render=VARIANT['eval_render'])
        env = env.unwrapped
    elif name == 'HalfCheetahEnv_cost':
        from envs.half_cheetah_cost import HalfCheetahEnv_cost as env
        # from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = env(args['reference'])
        # env = gym.make('DeskoCartpole-v0', des_v=args['reference'])
        #env = dreamer()
        env = env.unwrapped
    elif name == 'HalfCheetahEnv_cost_observation_noise':
        #from mujoco.half_cheetah import mujoco.HalfCheetahEnv as env
        from envs.half_cheetah_cost_observation_noise import HalfCheetahEnv_cost as env
        env = env(args['reference'])
        #env = dreamer()
        env = env.unwrapped
    elif name == 'HalfCheetahEnv_cost_process_noise':
        #from mujoco.half_cheetah import mujoco.HalfCheetahEnv as env
        from envs.half_cheetah_cost import HalfCheetahEnv_cost as env
        env = env(args['reference'])
        #env = dreamer()
        env = env.unwrapped
    elif name == 'Humanoid_stand_up':
        #from mujoco.half_cheetah import mujoco.HalfCheetahEnv as env
        from envs.humanoidstandup import HumanoidStandupEnv as env
        env = env()
        #env = dreamer()
        env = env.unwrapped
    else:
        env = gym.make(name)
        env = env.unwrapped
        if name == 'Quadrotorcons-v0':
            if 'CPO' not in VARIANT['algorithm_name']:
                env.modify_action_scale = False
        if 'Fetch' in name or 'Hand' in name:
            env.unwrapped.reward_type = 'dense'
    env.seed(SEED)
    return env


def get_model(name):
    if name == 'Koopman':
        from base_koopman_operator import base_Koopman as build_func
    elif name == 'Koopman_v2':
        from koopman_operator_v2 import Koopman as build_func
    elif name == 'Koopman_v5':
        from koopman_operator_v5 import Koopman as build_func
    elif name == 'Koopman_v10':
        from koopman_operator_V10 import Koopman as build_func
    elif name == 'Koopman_v10_meta':
        from koopman_operator_V10_meta_train import Koopman as build_func

    elif name == 'Koopman_v11':
        from koopman_operator_v11 import Koopman as build_func
    elif name == 'Koopman_v12':
        from koopman_operator_v12 import Koopman as build_func
    elif name == 'Koopman_v13':
        from koopman_operator_v13 import Koopman as build_func
    elif name == 'Koopman_v14':
        from koopman_operator_v14 import Koopman as build_func

    elif name == 'Koopman_v16':
        from koopman_operator_v16 import Koopman as build_func
    elif name == 'Koopman_v17':
        from koopman_operator_V17 import Koopman as build_func
    elif name == 'Koopman_v18':
        from koopman_operator_v18 import Koopman as build_func
    elif name == 'Koopman_meta_deterministic':
        from koopman_operator_meta import Koopman as build_func
    elif name == 'Koopman_meta_deterministic_tanh':
        from koopman_operator_meta_tanh import Koopman as build_func
    elif name == 'Koopman_meta':
        from koopman_operator_meta_V2 import Koopman as build_func
    elif name == 'Koopman_meta_v3':
        from koopman_operator_meta_V3 import Koopman as build_func
    elif name == 'Koopman_meta_v4':
        from koopman_operator_meta_V4 import Koopman as build_func
    elif name == 'Koopman_meta_v5':
        from koopman_operator_meta_V5 import Koopman as build_func
    elif name == 'Koopman_meta_v6':
        from koopman_operator_meta_V6 import Koopman as build_func
    elif name == 'Koopman_meta_VAE':
        from koopman_operator_meta_V9 import Koopman as build_func
    if name == 'MLP':
        from MLP import MLP as build_func
    return build_func

def get_replay_memory(name):
    if name == 'Koopman_meta_v3':
        rm = replay_memory_meta_simulator.ReplayMemoryWithPast
    elif name == 'Koopman_v10_meta':
        rm = replay_memory_meta_simulator.ReplayMemoryForDeSKO
    elif name == 'Koopman_meta_deterministic_tanh':
        rm = replay_memory_meta_simulator.ReplayMemoryTanh
    elif 'meta' in name:
        rm = replay_memory_meta_simulator.ReplayMemory
    else:
        rm = replay_memory.ReplayMemory

    return rm
def get_controller(model, args):
    if args['controller_name'] == 'MPC':
        # from controller import MPC as build_func
        # controller = build_func(model, args)
        from real_time_controller_v2 import real_time_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'MPC_dynamic':
        from real_time_controller_v2 import real_time_MPC_dynamic as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'MPC_with_motion_planning':
        from controller import MPC_with_motion_planning as build_func
        controller = build_func(model, args)
    # elif args['controller_name'] == 'Robust_fast_MPC':
    #     from real_time_controller import robust_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Time_varying_MPC':
        from controller import Time_varying_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Robust_fast_MPC':
        from real_time_controller_v2 import robust_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Robust_fast_dynamic_MPC':
        from real_time_controller_v2 import robust_MPC_dynamic_tracking as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Robust_fast_dynamic_MPC_partial_obs':
        from real_time_controller_v2 import robust_MPC_dynamic_tracking_partial_obs as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Robust_fast_dynamic_MPC_with_jerk':
        from real_time_controller_v2 import robust_MPC_dynamic_tracking_with_jerk_penalty as build_func
        controller = build_func(model, args)

    elif args['controller_name'] == 'Stochastic_MPC':
        from controller import Stochastic_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_v2':
        from controller import Stochastic_MPC_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_v3':
        from controller import Stochastic_MPC_v3 as build_func
        controller = build_func(model, args)

    elif args['controller_name'] == 'Stochastic_MPC_with_observation':
        from controller import Stochastic_MPC_with_observation as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_memory':
        from controller import Stochastic_MPC_with_memory as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_memory_v2':
        from controller import Stochastic_MPC_with_memory_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_observation_v2':
        from controller import Stochastic_MPC_with_observation_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Adaptive_MPC':
        from controller import Adaptive_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'osqp_Adaptive_MPC':
        from osqp_controller import Adaptive_MPC as build_func
    elif args['controller_name'] == 'Adaptive_MPC_v2':
        from controller import Adaptive_MPC_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Adaptive_MPC_v3':
        from controller import Adaptive_MPC_v3 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Adaptive_MPC_tanh':
        from controller import Adaptive_MPC_tanh as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_observation_v3':
        from controller import Stochastic_MPC_with_observation_v3 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_motion_planning':
        from controller import Stochastic_MPC_with_motion_planning as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'CEM_MPC':
        from CEM import CEM_MPC as build_func
        controller = build_func(model, args)
    else:
        print('controller does not exist')
        raise NotImplementedError
    return controller

def get_predictor(model, args):

    if args['predictor_name'] == 'AdaptivePredictor':
        from predictor import AdaptivePredictor as build_func
        predictor = build_func(model, args)
    elif args['predictor_name'] == 'StochasticAdaptivePredictor':
        from predictor import StochasticAdaptivePredictor as build_func
        predictor = build_func(model, args)
    else:
        print('predictor does not exist')
        raise NotImplementedError

    return predictor

def store_hyperparameters(path, args):
    np.save(path + "/hyperparameters.npy", args)


def restore_hyperparameters(path):
    args = np.load(path + "/hyperparameters.npy", allow_pickle=True).item()
    return args

def update_control_params(source_args, target_args):
    for key in ENV_PARAMS[source_args['env_name']].keys():
        if key =='num_of_task':
            continue
        target_args[key] = source_args[key]

    return target_args