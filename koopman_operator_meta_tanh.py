import tensorflow as tf
import numpy as np
import math
import os
import tensorflow_probability as tfp

from base_koopman_operator import base_Koopman
from utils import mlp

SCALE_DIAG_MIN_MAX = (-20, 2)
"""This version uses deterministic koopman operator with observation matrix"""
"""Kiwan: This version utilizes meta learning, for different task, we have different matrices respectively"""
class Koopman(base_Koopman):
    """Koopman.

    Attributes:
        A (tf.Variable): Weights of the Koopman operator
        B (tf.Variable): Weights of the Koopman operator
    """

    def __init__(
        self,
        args,
        **kwargs
    ):
        """
        Args:
            latent_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (function): The hidden layer activation function.

            output_activation (function, optional): The activation function used for
                the output layers. Defaults to tf.keras.activations.linear.

            name (str, optional): The Lyapunov critic name. Defaults to
                "lyapunov_critic".
        """



        # self.task_num = len(self.x_input)  # tell the number fo tasks from the shape of the imported list
        self.task_num = args['num_of_task']

        super(Koopman, self).__init__(args)

    def _create_place_holders(self, args):
        # list version
        self.x_input = tf.placeholder(tf.float32, [self.task_num, None, args['pred_horizon'], args['state_dim']], 'x')
        self.a_input = tf.placeholder(tf.float32, [self.task_num, None, args['pred_horizon'] - 1, args['act_dim']], 'a')

        self.shift = tf.Variable(np.zeros([args['state_dim']]), trainable=False, name="state_shift",
                                 dtype=tf.float32)
        self.scale = tf.Variable(np.zeros([args['state_dim']]), trainable=False, name="state_scale",
                                 dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros([args['act_dim']]), trainable=False, name="action_shift",
                                   dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros([args['act_dim']]), trainable=False, name="action_scale",
                                   dtype=tf.float32)

        self.loss_weight = tf.placeholder(tf.float32, [args['state_dim']], 'loss_weight')
        self.loss_weight_num = np.sqrt(np.diagonal(args['Q']))

        self.lr = tf.placeholder(tf.float32, None, 'learning_rate')
        self.l2_reg = tf.contrib.layers.l2_regularizer(args['l2_regularizer'])

    def _create_koopman_result_holder(self, args):
        # create multiple result holders
        # indicate how many tasks we have
        self.A_result = np.zeros([self.task_num, args['latent_dim'], args['latent_dim']])
        self.A_tensor = tf.Variable(self.A_result, trainable=False, name="A_tensor", dtype=tf.float32)

        self.B_result = np.zeros([self.task_num, args['act_dim'], args['latent_dim']])
        self.B_tensor = tf.Variable(self.B_result, trainable=False, name="B_tensor", dtype=tf.float32)

        self.C_result = np.zeros([self.task_num, args['latent_dim'], args['state_dim']])
        self.C_tensor = tf.Variable(self.C_result, trainable=False, name="C_tensor", dtype=tf.float32)

    def _create_encoder(self, args):


        '''
        Concatenate the pre-shuffled x_input and train the neural net first "BULK TRAINING" 

        a_input shape: (task_num, data_size, prediction_horizon-1, act_dim)
        x_input shape: (task_num, data_size, prediction_horizon, state_dim)
        mean, sigma shape: (task_num x data_size x pred_horizon x latent_dim)
        self.stochastic_latent shape: (n_of_random_seeds x task_num x data_size x pred_horizon x latent_dim)

        Here, the data size is the same for all tasks, and might not be consistent for all task
        Therefore, we need to concatenate the data in the 'data_size' dimension

        For matrix operations, we will separate the 'data_size' into 'sub_data_size' according to
        task_length for different tasks
        '''
        if args['activation'] == 'relu':
            activation = tf.nn.relu
        elif args['activation'] == 'elu':
            activation = tf.nn.elu
        self.mean = mlp(self.x_input, args['encoder_struct'] + [args['latent_dim']], activation, name='mean',
                        regularizer=self.l2_reg)


    def _create_koopman_operator(self, args):
        """
        Create the Koopman operators
        :param args:
        :return:

        """
        with tf.variable_scope('koopman', regularizer=self.l2_reg):
            self.A = tf.get_variable('A', shape=[self.task_num, args['latent_dim'], args['latent_dim']])
            self.B = tf.get_variable('B', shape=[self.task_num, args['act_dim'], args['latent_dim']])
            self.C = tf.get_variable('C', shape=[self.task_num, args['latent_dim'], args['state_dim']])

            self.mean_A = tf.reduce_mean(self.A, axis=0)
            self.mean_B = tf.reduce_mean(self.B, axis=0)
            self.mean_C = tf.reduce_mean(self.C, axis=0)
        return

    def _create_forward_pred(self, args):

        """
        Iteratively predict future state with the Koopman operator
        :param args(list):
        :return: forward_pred(Tensor): forward predictions
        Turn the predictions into lists containing lists

        self.stochastic_latent shape: (n_of_random_seeds x task_num x data_size x pred_horizon x latent_dim)
        mean, sigma shape: (task_num x data_size x pred_horizon x latent_dim)

        a_input shape: (task_num, data_size, prediction_horizon-1, act_dim)
        x_input shape: (task_num, data_size, prediction_horizon, state_dim)
        """
        forward_pred = []
        x_mean_forward_pred = []
        mean_forward_pred = []


        mean_t = self.mean[:, :, 0]


        for t in range(args['pred_horizon']-1):


            mean_t = tf.matmul(mean_t, self.A) + tf.matmul(self.a_input[:, :, t], self.B)
            x_mean_t = tf.matmul(mean_t, self.C)
            mean_forward_pred.append(mean_t)
            x_mean_forward_pred.append(x_mean_t)

        self.x_mean_forward_pred = tf.stack(x_mean_forward_pred, axis=2)
        self.mean_forward_pred = tf.stack(mean_forward_pred, axis=2)

        return

    def _create_optimizer(self, args):


        forward_pred_loss = tf.losses.mean_squared_error(labels=tf.stop_gradient(self.mean[:, :, 1:]), predictions=self.mean_forward_pred[:, :, :])


        # val_x = self.x_input[:, 1:]* self.scale + self.shift
        # val_y = self.x_mean_forward_pred[:, :] * self.scale + self.shift
        # self.val_loss = tf.abs(tf.reduce_mean((val_x - val_y)/(tf.abs(val_y)+1e-10)))

        val_x = self.x_input[:, :, 1:]
        val_y = self.x_mean_forward_pred[:, :, :]
        self.test_loss = self.reconstruct_loss = self.val_loss = tf.losses.mean_squared_error(labels=val_x, predictions=val_y)



        self.loss = self.reconstruct_loss + forward_pred_loss


        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=params)

        grad_norm = []
        self.grads = []
        for grad in tf.gradients(self.loss, params):
            if grad is not None:
                self.grads.append(grad)
                grad_norm.append(tf.norm(grad))
        grad_norm = tf.reduce_max(grad_norm)

        self.diagnotics.update({
            'loss': self.loss,
            'train_pred_error': self.val_loss,
            'gradient': grad_norm
        })
        self.opt_list.extend([self.train])

    def _create_prediction_model(self, args):

        self.x_t = tf.placeholder(tf.float32, [None, args['state_dim']], 'x_t')
        self.a_t = tf.placeholder(tf.float32, [None, args['pred_horizon']-1, args['act_dim']], 'a_t')
        self.task_to_draw = tf.placeholder(tf.int64)

        # self.shifted_x_t = (self.x_t - self.shift) / self.scale
        # self.shifted_a_t = (self.a_t - self.shift_u) / self.scale_u
        if args['activation'] == 'relu':
            activation = tf.nn.relu
        elif args['activation'] == 'elu':
            activation = tf.nn.elu
        else:
            print('activation function not implemented')
            raise NotImplementedError

        self.mean_t = mlp(self.x_t,
                           args['encoder_struct'] + [args['latent_dim']], activation, name='mean', reuse=True)

        forward_pred = []
        phi_t = self.mean_t
        for t in range(args['pred_horizon'] - 1):
            u = self.a_t[:, t]
            phi_t = tf.matmul(phi_t, self.A[self.task_to_draw]) + tf.matmul(u, self.B[self.task_to_draw])
            x_t = tf.matmul(phi_t, self.C[self.task_to_draw])
            forward_pred.append(x_t)
        self.future_states = tf.stack(forward_pred, axis=1)[:, :]


    def store_Koopman_operator(self, replay_memory):
        batch_dict = replay_memory.get_all_train_data()
        x = batch_dict['states']
        a = batch_dict['inputs']

        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a

        # Find loss and perform training operation
        feed_out = [self.A, self.B, self.C, tf.assign(self.A_tensor, self.A), tf.assign(self.B_tensor, self.B), tf.assign(self.C_tensor, self.C),]
        out = self.sess.run(feed_out, feed_in)
        self.A_result = out[0]
        self.B_result = out[1]
        self.C_result = out[2]

    def calc_val_loss(self, replay_memory):
        batch_dict = replay_memory.get_all_val_data()
        x = batch_dict['states']
        a = batch_dict['inputs']

        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a

        # Find loss
        feed_out = self.val_loss
        loss = self.sess.run(feed_out, feed_in)

        return loss

    def calc_test_loss(self, replay_memory):
        # batch_dict = replay_memory.get_all_test_data()
        batch_dict = replay_memory.get_all_test_data()
        x = batch_dict['states']
        a = batch_dict['inputs']

        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a

        # Find loss
        feed_out = self.val_loss
        loss = self.sess.run(feed_out, feed_in)

        return loss

    def learn(self, batch_dict, lr, args):
        x = batch_dict['states']
        a = batch_dict['inputs']

        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a
        feed_in[self.lr] = lr

        self.sess.run(self.opt_list, feed_in)

        diagnotics = self.sess.run([self.diagnotics[key] for key in self.diagnotics.keys()], feed_in)
        output = {}
        [output.update({key: value}) for (key, value) in zip(self.diagnotics.keys(), diagnotics)]
        for key in output.keys():
            if math.isnan(output[key]):
                print('NaN appears')
                raise ValueError
        return output

    def encode(self, x):
        feed_dict = {}

        feed_dict[self.x_t] = x
        # feed_dict[self.task_to_draw] = 0
        [mean] = self.sess.run([self.mean_t], feed_dict)

        return mean, None

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/model/')
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        feed_out = [self.A_tensor, self.B_tensor, self.C_tensor]
        out = self.sess.run(feed_out, {})
        self.A_result = out[0]
        self.B_result = out[1]
        self.C_result = out[2]
        success_load = True

        return success_load

    def make_prediction(self, x_t, u, task_num, args):
        feed_dict = {}
        future_states = []
        feed_dict[self.x_t] = x_t[0]
        feed_dict[self.a_t] = [u]
        feed_dict[self.task_to_draw] = task_num
        [future_states] = self.sess.run(self.future_states, feed_dict)

        return future_states

