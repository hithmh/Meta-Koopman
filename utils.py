import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
# from softtrunk_pybind_module import CurvatureCalculator


def visualize_predictions_meta_simulator(task_num, args, model, replay_memory, env, e=0):
    # modified version
    x = replay_memory.x_test_scenario
    u = replay_memory.u_test_scenario
    '''
    x_shape: task_num(list) x ep_steps     x state_dim
    u_shape: task_num(list) x ep_steps - 1 x action_dim
    '''
    furture_states = []
    t = args['history_horizon']
    plot_x_tick = range(x[task_num].shape[0])

    while t + args['pred_horizon'] < x[task_num].shape[0]:
        pred_trajectory = [x[task_num][t]]
        input = [x[task_num][t-args['history_horizon']:t+1]]
        # input.reverse()
        pred_trajectory.extend(model.make_prediction(input, u[task_num][t-args['history_horizon']:t + args['pred_horizon'] - 1], task_num, args))
        furture_states.append([
            plot_x_tick[t:t + args['pred_horizon']],
            np.array(pred_trajectory)*replay_memory.scale_x + replay_memory.shift_x])
        t += args['segment_of_test']
    x = x[task_num] * replay_memory.scale_x + replay_memory.shift_x
    plt.close()
    f, axs = plt.subplots(args['state_dim'] + args['act_dim'], sharex=True, figsize=(15, 15))

    for i in range(args['state_dim']):
        axs[i].plot(plot_x_tick, x[:, i], 'k')
        for obj in furture_states:
            axs[i].plot(obj[0], obj[1][:, i], 'r')

    for i in range(args['act_dim']):
        row = i + args['state_dim']
        axs[row].plot(plot_x_tick[:-1], u[task_num][:, i], 'b')

    plt.xlabel('Time Step')
    # plt.xlim([1, 2 * args['pred_horizon'] - 1])
    os.makedirs(args['log_path']+'/predictions', exist_ok=True)
    plt.savefig(args['log_path']+'/predictions/predictions_' + str(e) +'_task_num_' + str(task_num) + '.png')

def visualize_predictions_meta_simulator_tanh(task_num, args, model, replay_memory, env, e=0):
    # modified version
    x = replay_memory.x_test_scenario
    u = replay_memory.u_test_scenario
    '''
    x_shape: task_num(list) x ep_steps     x state_dim
    u_shape: task_num(list) x ep_steps - 1 x action_dim
    '''
    furture_states = []
    t = args['history_horizon']
    plot_x_tick = range(x[task_num].shape[0])

    while t + args['pred_horizon'] < x[task_num].shape[0]:
        pred_trajectory = [x[task_num][t]]
        input = [x[task_num][t-args['history_horizon']:t+1]]
        # input.reverse()
        pred_trajectory.extend(model.make_prediction(input, u[task_num][t-args['history_horizon']:t + args['pred_horizon'] - 1], task_num, args))
        furture_states.append([
            plot_x_tick[t:t + args['pred_horizon']],
            np.arctanh(np.array(pred_trajectory))*replay_memory.scale_x + replay_memory.shift_x])
        t += args['segment_of_test']
    x = np.arctanh(x[task_num]) * replay_memory.scale_x + replay_memory.shift_x
    plt.close()
    f, axs = plt.subplots(args['state_dim'] + args['act_dim'], sharex=True, figsize=(15, 15))

    for i in range(args['state_dim']):
        axs[i].plot(plot_x_tick, x[:, i], 'k')
        for obj in furture_states:
            axs[i].plot(obj[0], obj[1][:, i], 'r')

    for i in range(args['act_dim']):
        row = i + args['state_dim']
        axs[row].plot(plot_x_tick[:-1], u[task_num][:, i], 'b')

    plt.xlabel('Time Step')
    # plt.xlim([1, 2 * args['pred_horizon'] - 1])
    os.makedirs(args['log_path']+'/predictions', exist_ok=True)
    plt.savefig(args['log_path']+'/predictions/predictions_' + str(e) +'_task_num_' + str(task_num) + '.png')

def visualize_predictions_meta(task_num, args, model, replay_memory, env, e=0):
    # modified version
    x, u = replay_memory.get_test_trajectory_meta()
    x = np.asarray(x)
    u = np.asarray(u)
    '''
    x_shape: task_num x max_ep_steps x state_dim
    u_shape: task_num x max_ep_steps - 1 x action_dim
    '''
    furture_states = []
    t = args['history_horizon']
    plot_x_tick = range(x[0].shape[0])

    furture_states = []
    while t + args['pred_horizon'] < x[0].shape[0] + 1:
        pred_trajectory = x[:, t]
        pred_trajectory = np.reshape(pred_trajectory, [args['num_of_task'] + 1, 1, -1])

        input_list = []
        for j in range(args['num_of_task']+1):
            input = [x[j][t - i] for i in range(args['history_horizon'] + 1)]
            input.reverse()
            input_list.append(input)

        input_list = np.asarray(input_list)
        input_list = np.reshape(input_list, [args['num_of_task'] + 1, -1])

        prediction = model.make_prediction(input_list, u[:, t:t + args['pred_horizon'] - 1], args)
        pred_trajectory = np.concatenate((pred_trajectory, prediction), axis=1)
        pred_trajectory = pred_trajectory[task_num]
        furture_states.append([plot_x_tick[t:t + args['pred_horizon']], np.array(pred_trajectory)])
        t += args['segment_of_test']
        plt.close()

    f, axs = plt.subplots(args['state_dim'] + args['act_dim'], sharex=True, figsize=(15, 15))

    for i in range(args['state_dim']):
        axs[i].plot(plot_x_tick, x[task_num, :, i], 'k')
        for obj in furture_states:
            axs[i].plot(obj[0], obj[1][:, i], 'r')

    for i in range(args['act_dim']):
        row = i + args['state_dim']
        axs[row].plot(plot_x_tick[:-1], u[task_num, :, i], 'b')

    plt.xlabel('Time Step')
    # plt.xlim([1, 2 * args['pred_horizon'] - 1])
    os.makedirs(args['log_path']+'/predictions', exist_ok=True)
    plt.savefig(args['log_path']+'/predictions/predictions_' + str(e) +'_task_num_' + str(task_num) + '.png')


def visualize_predictions(args, model, replay_memory, env, e=0):
    """Plot predictions for a system against true time evolution
    Args:
        args: Various arguments and specifications
        sess: TensorFlow session
        net: Neural network dynamics model
        replay_memory: Object containing training/validation data
        env: Simulation environment
        e: Current training epoch
    """
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    # x = np.zeros((args['batch_size'], args['pred_horizon'], args['state_dim']), dtype=np.float32)
    # u = np.zeros((args['batch_size'], args['pred_horizon'] - 1, args['act_dim']), dtype=np.float32)
    # x = replay_memory.x_test
    # u = replay_memory.u_test
    '''x_shape'''
    x, u = replay_memory.get_test_trajectory()
    furture_states = []
    if hasattr(model, 'x_mean_forward_pred'):
        n = min(args['state_dim'], model.x_mean_forward_pred.shape[2])
    else:
        n = args['state_dim']
    # transform n into a number
    n = int(n)
    t = args['history_horizon']
    plot_x_tick = range(x.shape[0])

    while t + args['pred_horizon'] < x.shape[0]:
        pred_trajectory = [x[t, :n]]
        # input = [x[t - i] for i in range(args['history_horizon']+1)]
        # input.reverse()
        input = x[t-args['history_horizon']:t+1]
        pred_trajectory.extend(model.make_prediction(input, u[t-args['history_horizon']:t + args['pred_horizon']-1], args))
        furture_states.append([
            plot_x_tick[t:t + args['pred_horizon']],
            np.array(pred_trajectory)*replay_memory.scale_x[:n] + replay_memory.shift_x[:n]])
        t += args['segment_of_test']

    # preds = preds[1:]
    #
    # # Find mean, max, and min of predictions
    # pred_mean = np.mean(preds, axis=0)
    # pred_std = np.std(preds, axis=0)
    # pred_min = np.amin(preds, axis=0)
    # pred_max = np.amax(preds, axis=0)

    # diffs = np.linalg.norm(
    #     (preds[:, :args['pred_horizon']] - sess.run(net.shift)) / sess.run(net.scale) - x[0, :args['pred_horizon']], axis=(1, 2))
    # best_pred = np.argmin(diffs)
    # worst_pred = np.argmax(diffs)
    #
    # # Plot different quantities
    # x = x * sess.run(net.scale) + sess.run(net.shift)
    #
    # # # Find indices for random predicted trajectories to plot
    # ind0 = best_pred
    # ind1 = worst_pred
    #
    # Plot values
    plt.close()

    # plt.figure(figsize=(9, 6))
    f, axs = plt.subplots(n+args['act_dim'], sharex=True, figsize=(15, 15))
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')


    for i in range(n):
        axs[i].plot(plot_x_tick, x[:, i] * replay_memory.scale_x[i] +replay_memory.shift_x[i], 'k')
        for obj in furture_states:
            axs[i].plot(obj[0], obj[1][:, i], 'r')
        # axs[i].fill_between(range(1, 2 * args['pred_horizon']), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        # axs[i].set_ylim([np.amin(x[0, :, i]) - 0.2, np.amax(x[0, :, i]) + 0.2])3

    for i in range(args['act_dim']):
        row = i + n
        axs[row].plot(plot_x_tick[:-1], u[:, i], 'b')

    plt.xlabel('Time Step')
    # plt.xlim([1, 2 * args['pred_horizon'] - 1])
    os.makedirs(args['log_path']+'/predictions', exist_ok=True)
    plt.savefig(args['log_path']+'/predictions/predictions_' + str(e) + '.png')

def visualize_partial_predictions(args, model, replay_memory, env, e=0):
    """Plot predictions for a system against true time evolution
    Args:
        args: Various arguments and specifications
        sess: TensorFlow session
        net: Neural network dynamics model
        replay_memory: Object containing training/validation data
        env: Simulation environment
        e: Current training epoch
    """
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    # x = np.zeros((args['batch_size'], args['pred_horizon'], args['state_dim']), dtype=np.float32)
    # u = np.zeros((args['batch_size'], args['pred_horizon'] - 1, args['act_dim']), dtype=np.float32)
    # x = replay_memory.x_test
    # u = replay_memory.u_test
    x, u = replay_memory.get_test_trajectory()
    furture_states = []
    t = args['history_horizon']
    plot_x_tick = range(x.shape[0])

    while t + args['pred_horizon'] < x.shape[0]:
        pred_trajectory = [x[t, 6:9]]
        input = [x[t - i] for i in range(args['history_horizon']+1)]
        input.reverse()
        pred_trajectory.extend(model.make_prediction(input, u[t:t + args['pred_horizon']-1], args))
        furture_states.append([
            plot_x_tick[t:t + args['pred_horizon']],
            np.array(pred_trajectory)])
        t += args['segment_of_test']

    # preds = preds[1:]
    #
    # # Find mean, max, and min of predictions
    # pred_mean = np.mean(preds, axis=0)
    # pred_std = np.std(preds, axis=0)
    # pred_min = np.amin(preds, axis=0)
    # pred_max = np.amax(preds, axis=0)

    # diffs = np.linalg.norm(
    #     (preds[:, :args['pred_horizon']] - sess.run(net.shift)) / sess.run(net.scale) - x[0, :args['pred_horizon']], axis=(1, 2))
    # best_pred = np.argmin(diffs)
    # worst_pred = np.argmax(diffs)
    #
    # # Plot different quantities
    # x = x * sess.run(net.scale) + sess.run(net.shift)
    #
    # # # Find indices for random predicted trajectories to plot
    # ind0 = best_pred
    # ind1 = worst_pred
    #
    # Plot values
    plt.close()
    # plt.figure(figsize=(9, 6))
    reconstruct_dims = len(args['reconstruct_dims'])
    f, axs = plt.subplots(reconstruct_dims+args['act_dim'], sharex=True, figsize=(15, 15))
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')


    for i in range(reconstruct_dims):
        axs[i].plot(plot_x_tick, x[:, args['reconstruct_dims'][i]], 'k')
        for obj in furture_states:
            axs[i].plot(obj[0], obj[1][:, i], 'r')
        # axs[i].fill_between(range(1, 2 * args['pred_horizon']), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        # axs[i].set_ylim([np.amin(x[0, :, i]) - 0.2, np.amax(x[0, :, i]) + 0.2])3

    for i in range(args['act_dim']):
        row = i + reconstruct_dims
        axs[row].plot(plot_x_tick[:-1], u[:, i], 'b')

    plt.xlabel('Time Step')
    # plt.xlim([1, 2 * args['pred_horizon'] - 1])
    os.makedirs(args['log_path']+'/predictions', exist_ok=True)
    plt.savefig(args['log_path']+'/predictions/predictions_' + str(e) + '.png')

def mlp(input, sizes, activation, output_activation=None, name="", regularizer = None, reuse=None):
    """Creates a multi-layered perceptron using Tensorflow.

    Args:
        sizes (list): The size of each of the layers.

        activation (function): The activation function used for the
            hidden layers.

        output_activation (function, optional): The activation function used for the
            output layers. Defaults to tf.keras.activations.linear.

        regularizer (function, optional): Regularizer used to prevent overfitting

        name (st, optional): A nameprefix that is added before the layer name. Defaults
            to an empty string.

    Returns:
        output( Tensor): Output of the multi-layer perceptron
    """
    if reuse is None:
        trainable = True
    else:
        trainable = False

    with tf.variable_scope(name, reuse=reuse):
        # Create model
        for j in range(len(sizes) - 1):
            input = input if j == 0 else output
            if j < len(sizes) - 2:
                act = activation
            else:
                act = output_activation
            output = tf.layers.dense(
                    input,
                    sizes[j + 1],
                    activation=act,
                    name=name + "/l{}".format(j + 1),
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                    trainable=trainable
            )
            # output = tf.layers.dropout(output, rate=0.3)
    return output


def linear_encoder(input, size, input_shape, output_activation=None, name="", regularizer = None, reuse=None):
    """Creates a multi-layered perceptron using Tensorflow.

    Args:
        sizes (list): The size of each of the layers.

        activation (function): The activation function used for the
            hidden layers.

        output_activation (function, optional): The activation function used for the
            output layers. Defaults to tf.keras.activations.linear.

        regularizer (function, optional): Regularizer used to prevent overfitting

        name (st, optional): A nameprefix that is added before the layer name. Defaults
            to an empty string.

    Returns:
        output( Tensor): Output of the multi-layer perceptron
    """

    with tf.variable_scope(name, reuse=reuse):
        # Create model

        W = tf.get_variable('W', shape=[input_shape, size],)
        b = tf.get_variable('b', shape=[size])
        output = tf.matmul(input,  W) + b
    return output


def generate_reference_for_trunk(num_of_segment):
    threshold = [np.pi/6, np.pi]
    reference = []
    for n in range(num_of_segment):
        # theta = np.random.uniform(0, 1, 1) * threshold[0]
        # phi = np.random.uniform(-1, 1, 1) * threshold[1]
        theta = 1. * threshold[0]
        phi = 1. * threshold[1]
        reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))

    reference.append(np.zeros([2*num_of_segment], dtype=np.float32))
    return np.concatenate(reference, axis=0)

def generate_reference_for_trunkV2(num_of_segment):
    threshold_lowest = [0, 0]
    threshold_higher = [0.239,-0.127,-0.1]
    reference = []
    # for n in range(num_of_segment-1):
    #     # theta = np.random.uniform(0, 1, 1) * threshold[0]
    #     # phi = np.random.uniform(-1, 1, 1) * threshold[1]
    #     theta = 1. * threshold_lowest[0]
    #     phi = 1. * threshold_lowest[1]
    #     reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))
    reference.append(np.zeros([4*(num_of_segment-1)], dtype=np.float32))
    reference.append(np.squeeze(np.array(threshold_higher)))
    reference.append(np.zeros([3*(num_of_segment-1)], dtype=np.float32))
    return np.concatenate(reference, axis=0)

def generate_reference_for_trunkV3(num_of_segment):
    threshold_lowest = [0, 0]
    threshold_higher = [0.239,0.127,-0.00605]
    reference = []
    # for n in range(num_of_segment):
    #     # theta = np.random.uniform(0, 1, 1) * threshold[0]
    #     # phi = np.random.uniform(-1, 1, 1) * threshold[1]
    #     theta = 1. * threshold_lowest[0]
    #     phi = 1. * threshold_lowest[1]
    #     reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))
    reference.append(np.zeros([4*(num_of_segment)], dtype=np.float32))
    reference.append(np.squeeze(np.array(threshold_higher)))
    # reference.append(np.zeros([3*(num_of_segment)], dtype=np.float32))
    return np.concatenate(reference, axis=0)

def generate_reference_for_trunkV4(num_of_segment):
    angle = np.random.uniform(-np.pi, np.pi)
    # threshold_higher = np.array([0.23, 0.1 * np.cos(angle), 0.1 * np.sin(angle), ])
    threshold_higher = np.array([0.249, 0.08935, 0.07374])
    # for n in range(num_of_segment):
    #     # theta = np.random.uniform(0, 1, 1) * threshold[0]
    #     # phi = np.random.uniform(-1, 1, 1) * threshold[1]
    #     theta = 1. * threshold_lowest[0]
    #     phi = 1. * threshold_lowest[1]
    #     reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))
    reference = np.zeros(6*(num_of_segment))
    reference[6:9] = threshold_higher
    # reference.append(np.zeros([3*(num_of_segment)], dtype=np.float32))
    return reference



def generate_reference_for_trunkTK(num_of_segment):
    threshold_lowest = [0, 0]
    cc = CurvatureCalculator(CurvatureCalculator.SensorType.qualisys, "192.168.254.1")
    x,y,z = cc.get_object()
    threshold_higher = [x,y,z]
    reference = []
    # for n in range(num_of_segment-1):
    #     # theta = np.random.uniform(0, 1, 1) * threshold[0]
    #     # phi = np.random.uniform(-1, 1, 1) * threshold[1]
    #     theta = 1. * threshold_lowest[0]
    #     phi = 1. * threshold_lowest[1]
    #     reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))
    reference.append(np.zeros([4*(num_of_segment-1)], dtype=np.float32))
    reference.append(np.squeeze(np.array(threshold_higher)))
    reference.append(np.zeros([3*(num_of_segment-1)], dtype=np.float32))
    return np.concatenate(reference, axis=0)
# def generate_guidunce_for_trunk():
#     #get the x,y,z for the object ring
#     reference = []
#     cc = CurvatureCalculator(CurvatureCalculator.SensorType.qualisys, "192.168.254.1")
#     x,y,z = cc.get_object()
#     state = [x,y,z]
#     reference.append(state)
def linear_encoder(input, size, input_shape, output_activation=None, name="", regularizer=None, reuse=None):
        """Creates a multi-layered perceptron using Tensorflow.

        Args:
            sizes (list): The size of each of the layers.

            activation (function): The activation function used for the
                hidden layers.

            output_activation (function, optional): The activation function used for the
                output layers. Defaults to tf.keras.activations.linear.

            regularizer (function, optional): Regularizer used to prevent overfitting

            name (st, optional): A nameprefix that is added before the layer name. Defaults
                to an empty string.

        Returns:
            output( Tensor): Output of the multi-layer perceptron
        """

        with tf.variable_scope(name, reuse=reuse):
            # Create model

            W = tf.get_variable('W', shape=[input_shape, size], )
            b = tf.get_variable('b', shape=[size])
            output = tf.matmul(input, W) + b
        return output

