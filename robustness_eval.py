import tensorflow as tf
import os
from variant import *
import numpy as np
import time
import logger
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def get_distrubance_function(args, env, model):
    env_name = args['env_name']
    if 'trunk_arm_sim' in env_name:
        disturbance_step = trunk_sim_disturbance_step(args, env)
    else:
        if args['evaluation_form']=='nominal_model':
            disturbance_step = model_step(args, env, model)
        else:
            disturbance_step = base_disturbance_step(args, env)
        # disturbance_step = None

    return disturbance_step


class base_disturbance_step(object):

    def __init__(self, args, env):
        self.form_of_eval = args['evaluation_form']
        self.time = 0
        self.env = env
        self.args = args
        self.initial_pos = np.random.uniform(0., np.pi, size=[args['act_dim']])

    def step_halfcheetah(self, action, s, reference):

        if self.form_of_eval == 'impulse':
            s_, r, done, info = self.impulse(action, s)
        elif self.form_of_eval == 'constant_impulse':
            s_, r, done, info = self.constant_impulse(action, s)
        elif self.form_of_eval == 'various_disturbance':
            s_, r, done, info = self.various_disturbance(action, s)
        else:
            s_, r, done, info = self.normal_step_halfcheetah(action, reference)
        self.time += 1

        return s_, r, done, info

    def step(self, action, s):

        if self.form_of_eval == 'impulse':
            s_, r, done, info = self.impulse(action, s)
        elif self.form_of_eval == 'constant_impulse':
            s_, r, done, info = self.constant_impulse(action, s)
        elif self.form_of_eval == 'various_disturbance':
            s_, r, done, info = self.various_disturbance(action, s)
        else:
            s_, r, done, info = self.normal_step(action)
        self.time += 1

        return s_, r, done, info

    def impulse(self, action, s):

        if self.time == self.args['impulse_instant']:
            d = self.args['magnitude'] * np.sign(s[0])
        else:
            d = 0
        s_, r, done, info = self.env.step(action, impulse=d)

        return s_, r, done, info

    def constant_impulse(self, action, s):
        if self.time % self.args['impulse_instant']==0:
            d = self.args['magnitude'] * np.sign(s[0])
        else:
            d = 0
        s_, r, done, info = self.env.step(action, d)
        return s_, r, done, info

    def various_disturbance(self, action, s):
        if self.args['form'] == 'sin':
            d = np.sin(2 *np.pi /self.args['period'] * self.time + self.initial_pos) * self.args['magnitude']
        s_, r, done, info = self.env.step(action, impulse=d)
        return s_, r, done, info

    def normal_step(self, action):
        s_, r, done, info = self.env.step(action)
        return s_, r, done, info

    def normal_step_halfcheetah(self, action, reference):
        s_, r, done, info = self.env.step(action)
        return s_, r, done, info

    def reset(self):
        self.t = 0
        # self.env.randomize()
        s = self.env.reset()

        return s

class model_step(base_disturbance_step):

    def __init__(self, args, env, model):
        super(model_step, self).__init__(args, env)
        self.model = model
        self.history_horizon = model.history_length


    def step(self, action, s):
        states, actions = self.get_state_and_action(action)
        s_= self.model.rollout(states, actions)
        r = 0
        self.t += 1
        self.state_memory.append(s_)
        self.action_memory.append(action)

        return s_, r, False, {}

    def get_state_and_action(self, action):
        if self.t>=self.history_horizon:
            state = np.concatenate([self.state_memory[-self.history_horizon:], [self.s]])
            action = np.concatenate([self.action_memory[-self.history_horizon+1:], [action]])
        else:
            delta_t = self.history_horizon - self.t

            if self.t == 0:
                action = np.tile(action, [delta_t, 1])
                state = np.tile(self.s, [delta_t+1, 1])
            else:
                state = np.concatenate(
                    [self.state_memory[-(self.history_horizon - delta_t):], np.tile(self.s, [delta_t+1, 1])])
                action = np.concatenate([self.action_memory[-(self.history_horizon-delta_t):], np.tile(action, [delta_t, 1])])

        return state, action

    def reset(self):
        self.s = self.env.reset()
        self.state_memory = [self.s]
        self.action_memory = []
        self.t = 0

        return self.s


class trunk_sim_disturbance_step(base_disturbance_step):

    def __init__(self, args, env):
        super(trunk_sim_disturbance_step, self).__init__(args, env)

    def impulse(self, action, s):

        if self.time == self.args['impulse_instant']:
            d = self.args['magnitude'] * -np.sign(action)
        else:
            d = 0
        s_, r, done, info = self.env.step(action, impulse=d)
        return s_, r, done, info

    def constant_impulse(self, action, s):
        if self.time % self.args['impulse_instant']==0:
            d = self.args['magnitude'] * -np.sign(action)
        else:
            d = 0
        s_, r, done, info = self.env.step(action, d)
        return s_, r, done, info



def constant_impulse(policy, env, variant):


    log_path = variant['log_path'] + '/eval/constant_impulse'
    variant.update({'magnitude': 0})
    logger.configure(dir=log_path, format_strs=['csv'])
    for magnitude in variant['magnitude_range']:
        variant['magnitude'] = magnitude
        diagnostic_dict, _ = evaluation(variant, env, policy, verbose=False)

        string_to_print = ['magnitude', ':', str(magnitude), '|']
        [string_to_print.extend([key, ':', str(round(diagnostic_dict[key], 2)), '|'])
         for key in diagnostic_dict.keys()]
        print(''.join(string_to_print))

        logger.logkv('magnitude', magnitude)
        [logger.logkv(key, diagnostic_dict[key]) for key in diagnostic_dict.keys()]
        logger.dumpkvs()


def various_disturbance(policy, env, variant):

    log_path = variant['log_path'] + '/eval/various_disturbance-' + variant['form']
    variant.update({'period': 0})
    logger.configure(dir=log_path, format_strs=['csv'])
    for period in variant['period_list']:
        variant['period'] = period
        diagnostic_dict, _ = evaluation(variant, env, policy, verbose=False)
        frequency = 1. / period
        string_to_print = ['frequency', ':', str(frequency), '|']
        [string_to_print.extend([key, ':', str(round(diagnostic_dict[key], 2)), '|'])
         for key in diagnostic_dict.keys()]
        print(''.join(string_to_print))

        logger.logkv('frequency', frequency)
        [logger.logkv(key, diagnostic_dict[key]) for key in diagnostic_dict.keys()]
        logger.dumpkvs()


def instant_impulse(policy, env, variant):

    log_path = variant['log_path'] + '/eval/impulse'
    variant.update({'magnitude': 0})
    logger.configure(dir=log_path, format_strs=['csv'])
    for magnitude in  variant['magnitude_range']:
        variant['magnitude'] = magnitude
        diagnostic_dict, _ = evaluation(variant, env, policy)

        string_to_print = ['magnitude', ':', str(magnitude), '|']
        [string_to_print.extend([key, ':', str(round(diagnostic_dict[key], 2)), '|'])
         for key in diagnostic_dict.keys()]
        print(''.join(string_to_print))

        logger.logkv('magnitude', magnitude)
        [logger.logkv(key, diagnostic_dict[key]) for key in diagnostic_dict.keys()]
        logger.dumpkvs()


def dynamic(policy, env, root_variant, variant):

    log_path = root_variant['log_path'] + '/eval/dynamic/'+root_variant['eval_additional_description']
    root_variant.update({'magnitude': 0})
    logger.configure(dir=log_path, format_strs=['csv'])

    diagnotics, paths = evaluation(root_variant, env, policy)

    root_variant['state_dim'] = variant['reference'].shape[0]

    max_len = 0
    if root_variant['env_name'] == 'trunk_arm_sim':
        for path in paths['c']:
            path_length = len(path)
            if path_length > max_len:
                max_len = path_length
    else:
        for path in paths['s']:
            path_length = len(path)
            if path_length > max_len:
                max_len = path_length
    if 's' in paths.keys():
        average_path = np.average(np.array(paths['s']), axis=0)
        std_path = np.std(np.array(paths['s']), axis=0)

    if 'a' in paths.keys():
        action_average_path = np.average(np.array(paths['a']), axis=0)
        action_std_path = np.std(np.array(paths['a']), axis=0)
    tracking_error = np.average(np.array(paths['c']), axis=0)
    # tracking_error_single = np.array(paths['c'])
    tracking_error_std = np.std(np.array(paths['c']), axis=0)


    reference_half_cheetah = np.zeros(tracking_error.shape)

    for i in range(max_len):
        if 'HalfCheetahEnv_cost' in root_variant['env_name'] or 'trunk_arm_sim' in root_variant['env_name']:

            logger.logkv('tracking_error', np.around(tracking_error[i], 3))
            logger.logkv('tracking_error_std', np.around(tracking_error_std[i], 3))

        else:
            logger.logkv('average_path', np.around(average_path[i], 8))
            logger.logkv('average_path_std', np.around(std_path[i], 8))
            logger.logkv('tracking_error', np.around(tracking_error[i], 8))
            logger.logkv('tracking_error_std', np.around(tracking_error_std[i], 8))

        if 'a' in paths.keys():
            logger.logkv('action_average_path', np.around(action_average_path[i], 8))
            logger.logkv('action_average_path_std', np.around(action_std_path[i], 8))

        if 'HalfCheetahEnv_cost' not in root_variant['env_name']:
            if 'reference' in paths.keys():
                logger.logkv('reference', np.around(paths['reference'][0][i],3))
        else:
            logger.logkv('reference', np.around(reference_half_cheetah[i],3))

        logger.logkv('t', i+1)
        # logger.logkv('reference', paths['reference'][0][i])
        logger.dumpkvs()
    if root_variant['directly_show']:
        fig, ax = plt.subplots(root_variant['state_dim'], sharex=True, figsize=(15, 15))

        if root_variant['plot_average']:
            t = range(max_len)
            for i in range(root_variant['state_dim']):
                ax[i].plot(t, average_path[:,i], color='red')
                # if env_name =='cartpole_cost':
                #     ax.fill_between(t, (average_path - std_path)[:, 0], (average_path + std_path)[:, 0],
                #                     color='red', alpha=.1)
                # else:
                ax[i].fill_between(t, average_path[:, i]-std_path[:,i], average_path[:,i]+std_path[:,i], color='red', alpha=.1)
                if 'reference' in paths.keys():
                    for path in paths['reference']:
                        path = np.array(path)
                        path_length = len(path)
                        if path_length == max_len:
                            t = range(path_length)

                            ax[i].plot(t, path[:, i], color='brown', linestyle='dashed', label='reference')
                            break
                        else:
                            continue
        else:
            for i in range(root_variant['state_dim']):
                for path in paths['s']:
                    path_length = len(path)
                    t = range(path_length)
                    path = np.array(path)
                    ax[i].plot(t, path[:, i], color='red')
                    if path_length>max_len:
                        max_len = path_length

                if 'reference' in paths.keys():
                    for path in paths['reference']:
                        path = np.array(path)
                        path_length = len(path)
                        if path_length == max_len:
                            t = range(path_length)

                            ax[i].plot(t, path[:, i], color='brown', linestyle='dashed', label='reference')
                            break
                        else:
                            continue
                handles, labels = ax[i].get_legend_handles_labels()
                ax[i].legend(handles, labels, fontsize=20, loc=2, fancybox=False, shadow=False)
        plt.savefig('-'.join([root_variant['env_name'], variant['alg_name'], variant['controller_name'], 'dynamic-state.pdf']))
        plt.show()

        # # plot actions
        # fig, ax = plt.subplots(root_variant['act_dim'], sharex=True, figsize=(15, 15))
        #
        # if root_variant['plot_average']:
        #     t = range(max_len)
        #     for i in range(root_variant['act_dim']):
        #         ax[i].plot(t, action_average_path[:, i], color='red')
        #         # if env_name =='cartpole_cost':
        #         #     ax.fill_between(t, (average_path - std_path)[:, 0], (average_path + std_path)[:, 0],
        #         #                     color='red', alpha=.1)
        #         # else:
        #         ax[i].fill_between(t, action_average_path[:, i] - action_std_path[:, i], action_average_path[:, i] + action_std_path[:, i],
        #                            color='red', alpha=.1)
        # else:
        #     for i in range(root_variant['act_dim']):
        #         for path in paths['a']:
        #             path_length = len(path)
        #             t = range(path_length)
        #             path = np.array(path)
        #             ax[i].plot(t, path[:, i], color='red')
        #             if path_length > max_len:
        #                 max_len = path_length
        #
        #         handles, labels = ax[i].get_legend_handles_labels()
        #         ax[i].legend(handles, labels, fontsize=20, loc=2, fancybox=False, shadow=False)
        # plt.savefig(
        #     '-'.join([root_variant['env_name'], variant['alg_name'], variant['controller_name'], 'dynamic-action.pdf']))
        # plt.show()

        if 'c' in paths.keys():
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111)
            for path in paths['c']:
                t = range(len(path))
                ax.plot(t, path)
            plt.savefig('-'.join([root_variant['env_name'], variant['alg_name'], variant['controller_name'], 'dynamic-cost.pdf']))
            plt.show()
        return


def simple_validation(controller, env, args):
    s = env.observation_space.sample()
    if args['env_name'] != 'linear_sys':
        s = controller.model.encode([s])
    controller.check_controllability()
    path = []
    control_history = []
    for i in range(args['max_ep_steps']):
        # a = controller.choose_action(s[:controller.state_dim])
        a = controller.simple_choose_action(s)
        s_ = controller.linear_predict(s, a)
        path.append(s[:controller.state_dim])
        control_history.append(a)
        s = s_

    path = np.array(path)
    control_history = np.array(control_history)
    f, axs = plt.subplots(args['state_dim'] + args['act_dim'], sharex=True, figsize=(15, 15))

    plot_x_tick = range(path.shape[0])
    plot_a_tick = range(control_history.shape[0])
    for i in range(args['state_dim']):
        axs[i].plot(plot_x_tick, path[:, i], 'k')
    axs[-1].plot(plot_a_tick, control_history, 'k')
    plt.show()
    print('rollout_finished')


def param_variation(policy, env, variant):

    params = env.get_list_of_params()

    param_name = env.get_parameter_name()
    log_path = variant['log_path'] + '/eval'

    log_path = log_path + '/param_variation'
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('num_of_paths', variant['num_of_paths'])

    for j, param in enumerate(params):
        env.set_parameter(param)
        diagnostic_dict, paths = evaluation(variant, env, policy, verbose=False)
        np.save(log_path + '/state_path_' + str(j) + '.npy', np.array(paths['s']))
        np.save(log_path + '/cost_path_' + str(j) + '.npy', np.array(paths['c']))
        np.save(log_path + '/action_path_' + str(j) + '.npy', np.array(paths['a']))
        np.save(log_path + '/reference_path_' + str(j) + '.npy', np.array(paths['reference']))
        string_to_print = []
        [string_to_print.extend([key, ':', str(round(diagnostic_dict[key], 2)), '|'])
         for key in diagnostic_dict.keys()]
        print(''.join(string_to_print))

        for i, name in enumerate(param_name):
            logger.logkv(name, param[i])
        logger.logkv('index', str(j))

        [logger.logkv(key, diagnostic_dict[key]) for key in diagnostic_dict.keys()]
        logger.dumpkvs()


def evaluation(variant, env, policy, verbose=True):
    env_name = variant['env_name']

    # disturbance_step = get_distrubance_function(env_name)

    max_ep_steps = variant['max_ep_steps']

    a_dim = env.action_space.shape[0]
    # For analyse
    Render = variant['eval_render']

    # Training setting

    total_cost = []
    death_rates = []
    form_of_eval = variant['evaluation_form']
    trial_list = os.listdir(variant['log_path'])
    episode_length = []
    cost_paths = []
    value_paths = []
    state_paths = []
    ref_paths = []
    action_paths = []
    variant['trials_for_eval'].append('model')
    count = 0
    for trial in trial_list:
        if trial in variant['trials_for_eval']:
            count += 1
    total_iteration = int(np.ceil(variant['num_of_paths']/count))
    for trial in trial_list:

        # if trial == 'eval':
        #     continue
        if trial not in variant['trials_for_eval']:
            continue
        success_load = policy.restore(os.path.join(variant['log_path'], trial))

        if not success_load:
            continue
        disturbance_step = get_distrubance_function(variant, env, policy.model)
        die_count = 0
        seed_average_cost = []
        for i in range(total_iteration):
            path = []
            state_path = []
            value_path = []
            ref_path = []
            act_path = []
            cost = 0
            policy.reset()
            s = disturbance_step.reset()
            if 'Fetch' in env_name or 'Hand' in env_name:
                s = np.concatenate([s[key] for key in s.keys()])


            for j in range(max_ep_steps):

                if Render:
                    env.render()
                if 'HalfCheetah' in variant['env_name']:

                    action = policy.choose_action(s, variant['reference'][j:j + variant['pred_horizon'], :], True)
                else:
                    action = policy.choose_action(s, variant['reference'], True)
                act_path.append(action)

                if 'HalfCheetah' in variant['env_name']:
                    s_, r, done, info = disturbance_step.step_halfcheetah(action, s, variant['reference'][j])
                else:
                    s_, r, done, info = disturbance_step.step(action, s)
                # value_path.append(policy.evaluate_value(s,a))
                done = False if variant['evaluation_form'] == 'dynamic'  else done
                path.append(r)
                cost += r
                if 'Fetch' in env_name or 'Hand' in env_name:
                    s_ = np.concatenate([s_[key] for key in s_.keys()])
                if 'reference' in info.keys():
                    ref_path.append(info['reference'])
                if 'Humanoid' in variant['env_name']:
                    state_path.append(s[0])
                else:
                    state_path.append(s)
                if j == max_ep_steps - 1:
                    done = True
                s = s_
                if done:

                    seed_average_cost.append(cost)
                    episode_length.append(j)
                    if j < max_ep_steps-1:
                        die_count += 1
                    break

            cost_paths.append(path)
            value_paths.append(value_path)
            state_paths.append(state_path)
            ref_paths.append(ref_path)
            action_paths.append(act_path)

        death_rates.append(die_count/(i+1)*100)
        total_cost.append(np.mean(seed_average_cost))

    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)
    death_rate = np.mean(death_rates)
    death_rate_std = np.std(death_rates, axis=0)
    average_length = np.average(episode_length)

    diagnostic = {'return': total_cost_mean,
                  'return_std': total_cost_std,
                  'death_rate': death_rate,
                  'death_rate_std': death_rate_std,
                  'average_length': average_length}


    if verbose:
        string_to_print = []
        [string_to_print.extend([key, ':', str(diagnostic[key]), '|'])
         for key in diagnostic.keys()]
        print('######################################################')
        print(''.join(string_to_print))
        print('######################################################')

    path_dict = {'c': cost_paths, 'v':value_paths}
    if 'reference' in info.keys():
        path_dict.update({'reference': ref_paths})

    path_dict.update({'s': state_paths,
                      'a': action_paths})

    return diagnostic, path_dict


def main():
    root_args = VARIANT
    env = get_env_from_name(root_args)
    for name in VARIANT['eval_list']:
        args = restore_hyperparameters('/'.join(['./log', VARIANT['env_name'], name]))
        args['s_bound_low'] = env.observation_space.low
        args['s_bound_high'] = env.observation_space.high
        args['a_bound_low'] = env.action_space.low
        args['a_bound_high'] = env.action_space.high

        root_args['s_bound_low'] = env.observation_space.low
        root_args['s_bound_high'] = env.observation_space.high
        root_args['a_bound_low'] = env.action_space.low
        root_args['a_bound_high'] = env.action_space.high

        if 'Fetch' in VARIANT['env_name'] or 'Hand' in VARIANT['env_name']:
            args['state_dim'] = env.observation_space.spaces['observation'].shape[0] \
                    + env.observation_space.spaces['achieved_goal'].shape[0] + \
                    env.observation_space.spaces['desired_goal'].shape[0]
        else:
            args['state_dim'] = env.observation_space.shape[0]
            root_args['state_dim'] = env.observation_space.shape[0]
        args = update_control_params(root_args, args)

        args['act_dim'] = env.action_space.shape[0]
        root_args['act_dim'] = env.action_space.shape[0]

        build_func = get_model(args['alg_name'])
        model = build_func(args)
        controller = get_controller(model, args)
        root_args['log_path'] = '/'.join(['./log', VARIANT['env_name'], name])
        if root_args['evaluation_form'] == 'dynamic' or root_args['evaluation_form'] == 'nominal_model':
            dynamic(controller, env, root_args, args)
        elif root_args['evaluation_form'] == 'constant_impulse':
            constant_impulse(controller, env, root_args)
        elif root_args['evaluation_form'] == 'impulse':
            instant_impulse(controller, env, root_args)
        elif root_args['evaluation_form'] == 'various_disturbance':
            various_disturbance(controller, env, root_args)
        elif root_args['evaluation_form'] == 'param_variation':
            param_variation(controller, env, root_args)
        else:
            print('The evaluation function '+ root_args['evaluation_form'] +' does not exist')
        tf.reset_default_graph()



if __name__ == '__main__':
    main()