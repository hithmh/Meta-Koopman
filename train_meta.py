

from variant import *
from utils import visualize_predictions_meta_simulator, visualize_partial_predictions, visualize_predictions_meta_simulator_tanh

import logger
from robustness_eval import *
import time

'''
meta training file
'''

def main():
    args = VARIANT
    root_dir = args['log_path']
    env = get_env_from_name(args)
    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['s_bound_low'] = env.observation_space.low
    args['s_bound_high'] = env.observation_space.high
    args['a_bound_low'] = env.action_space.low
    args['a_bound_high'] = env.action_space.high
    os.makedirs(root_dir, exist_ok=True)

    if args['train_model']:
        store_hyperparameters(root_dir, args)   # store hyperparameters into a file

    for i in range(args['start_of_trial'], args['start_of_trial'] + args['num_of_trials']):

        args['log_path'] = root_dir + '/' + str(i)
        print('logging to ' + args['log_path'])

        model = train(args, env)

        if args['eval_control']:
            args['log_path'] = root_dir
            if args['store_hyperparameter']:
                store_hyperparameters(root_dir, args)

            controller = get_controller(model, args)
            controller._build_controller()
            controller.check_controllability()

            if args['evaluation_form'] == 'dynamic':
                dynamic(controller, env, args, args)
            elif args['evaluation_form'] == 'constant_impulse':
                constant_impulse(controller, env, args)
            # simple_validation(controller, env, args)
        tf.reset_default_graph()

def train(args, env):

    build_func = get_model(args['alg_name'])
    model = build_func(args)

    if args['train_model'] is False:
        if args['env_name'] == 'linear_sys':
            model.A_result = env.A.T
            model.B_result = env.B.T
        else:
            success = model.restore(args['log_path'])
            if not success:
                print(args['log_path'] + ' does not exist')
                raise NotImplementedError
        return model

    if args['continue_training']:
        success = model.restore(args['log_path'])
        if not success:
            print(args['log_path'] + ' does not exist')
            raise NotImplementedError

    logger.configure(dir=args['log_path'], format_strs=['csv'])

    [shift, scale, shift_u, scale_u] = model.get_shift_and_scale()  # create placeholders
    # Generate training data
    rm_build_func = get_replay_memory(args['alg_name'])
    replay_memory = rm_build_func(args, shift, scale, shift_u, scale_u, env, predict_evolution=True)
    model.set_shift_and_scale(replay_memory)

    # save the sampled parameters of replay_memory into a file under log_path
    np.save(args['log_path'] + '/sampled_parameters.npy', replay_memory.sampled_parameters)

    # Define counting variables
    count_decay = 0
    decay_epochs = []

    # Initialize variable to track validation score over time
    old_score = 1e20

    lr = args['learning_rate']
    graph_loop_counter = 0  # for graph loop
    for e in range(args['num_epochs']):
        # Initialize loss
        loss = 0.0
        val_loss = 0.0
        loss_count = 0
        b = 0
        replay_memory.reset_batchptr_train()

        # Loop over batches
        while b < np.amin(replay_memory.n_batches_train):

            # Get inputs
            batch_dict = replay_memory.next_batch_train_meta()
            out = model.learn(batch_dict, lr, args)
            b += 1

        model.store_Koopman_operator(replay_memory)
        # Evaluate loss on validation set
        score_val = model.calc_val_loss(replay_memory)
        score_test = model.calc_test_loss(replay_memory)
        # print(tf.shape(model.mean))
        for key in out.keys():
            counter=0
            if key == 'seperate_loss':
                for i in range(len(out[key])):
                    logger.logkv(key+str(counter), out[key][i])
                    counter += 1
            else:
                logger.logkv(key, out[key])

        # logger.logkv('train_loss', loss)
        logger.logkv('epoch', e)
        logger.logkv('validation_loss', score_val)
        logger.logkv('test_loss', score_test)
        logger.logkv('learning_rate', lr)

        logger.dumpkvs()
        string_to_print = [args['alg_name'] + args['additional_description'], '|']
        string_to_print.extend(['epoch:', str(e), '|'])
        for key in out.keys():
            if key == 'seperate_loss':
                continue
            else:
                string_to_print.extend([key, ':', str(round(out[key], 2)), '|'])
        # [string_to_print.extend([key, ':', str(round(out[key], 2)), '|']) for key in out.keys()]
        string_to_print.extend(['validation_loss:', str(round(score_val, 8)), '|'])
        string_to_print.extend(['test_loss:', str(round(score_test, 8)), '|'])
        string_to_print.extend(['learning_rate:', str(round(lr, 8)), '|'])
        print(''.join(string_to_print))
        # print('Validation Loss: {0:f}'.format(score))

        # Set learning rate
        if (old_score - score_val) < -0.01 and e >= 8:
            count_decay += 1
            decay_epochs.append(e)
            # if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2:
            #     break

            # lr = args['learning_rate'] * (args['decay_rate'] ** count_decay)
            # print('setting learning rate to ', lr)
        ## stair decay
        if (e + 1) % args['decay_steps'] == 0:
            lr = lr * args['decay_rate']
        # ## constant decay
        # frac = 1.0 - e / args['num_epochs']
        # lr = args['learning_rate'] * frac
        # print('setting learning rate to ', lr)

        old_score = score_val
        if e % args['save_frequency'] == 0:
            loop_end = args['num_of_task']
            graph_loop_counter = graph_loop_counter
            model.save_result(args['log_path'], verbose=False )
            # print("model saved to {}".format(args['log_path']))
            # if 'reconstruct_dims' in args.keys():
            #     visualize_partial_predictions(args, model, replay_memory, env, e)
            # else:
            if 'tanh' in args['alg_name']:
                visualize_predictions_meta_simulator_tanh(graph_loop_counter, args, model, replay_memory, env, e)
            else:
                visualize_predictions_meta_simulator(graph_loop_counter, args, model, replay_memory, env, e)
            graph_loop_counter = (graph_loop_counter+1) % loop_end
    return model



if __name__ == '__main__':
    main()