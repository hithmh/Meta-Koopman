import math
import numpy as np
import random
import progressbar
import os


# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, shift, scale, shift_u, scale_u, env, predict_evolution=False):
        """Constructs object to hold and update training/validation data.
        Args:
            args: Various arguments and specifications
            shift: Shift of state values for normalization
            scale: Scaling of state values for normalization
            shift_u: Shift of action values for normalization
            scale_u: Scaling of action values for normalization
            env: Simulation environment
            net: Neural network dynamics model
            sess: TensorFlow session
            predict_evolution: Whether to predict how system will evolve in time

            TODO: check files
        """
        self._configure(args, shift, scale, shift_u, scale_u, env)

        print('validation fraction: ', args['val_frac'])

        if args['import_saved_data'] or args['continue_data_collection']:

            # add a task num to indicate task to read
            self._restore_data(task_num=1, path='./data/' + args['env_name'])
            self._process_data(args)

            print('creating splits...')
            self._create_split(args)
            self._determine_shift_and_scale(args)
            if args['continue_data_collection']:
                self.total_steps = self.x.shape[0]
                self._generate_data(args)
                self._process_data(args)
                self._create_split(args)
                self._determine_shift_and_scale(args)
                self._shift_scale(args)

        elif args['import_merging_saved_data']:
            for i in range(self.task_num):
                print("randomizing, trial_" + str(i + 1))
                env.randomize()

                print("generating data..., trial_" + str(i + 1))
                self._generate_data(args)
                self._process_data(args)

                print("creating splits..., trial_" + str(i + 1))
                self._create_split(args)

                print("shifting/scaling data..., trial_" + str(i + 1))
                self._shift_scale(args)
                self._store_data_in_holder()
            self._transfer()
            print("merging data ...")
            self._merging_data()  # Finished-Checked
            self._process_data_meta(args)  # Finished-Checked

            print('creating splits...')
            self._create_split_meta(args)  # Finished
            self._determine_shift_and_scale_meta(args)

        else:
            # TODO: data size should be a clever list [Task_0 size, Task_1 size ...]

            for i in range(self.task_num):
                print("randomizing, trial_" + str(i + 1))
                env.randomize()

                print("generating data..., trial_" + str(i + 1))
                self._generate_data(args)
                self._process_data(args)

                print("creating splits..., trial_" + str(i + 1))
                self._create_split(args)

                print("shifting/scaling data..., trial_" + str(i + 1))
                self._shift_scale(args)

                print("saving data..., trial_" + str(i + 1))
                self.save_data('./data/' + args['env_name'])

    def _configure(self, args, shift, scale, shift_u, scale_u, env):
        self.batch_size = args['batch_size']
        self.seq_length = args['pred_horizon']
        self.shift_x = shift
        self.scale_x = scale
        self.shift_u = shift_u
        self.scale_u = scale_u
        self.env = env
        self.total_steps = 0
        self.task_num = args['num_of_task']

        self.x_test_holder = []
        self.u_test_holder = []
        self.test_length_list_holder = []
        self.length_list_holder = []
        self.x_raw_holder = []
        self.u_raw_holder = []

    def _store_data_in_holder(self):
        self.x_raw_holder.append(self.x_raw)
        self.x_test_holder.append(self.x_test)
        self.u_test_holder.append(self.u_test)
        self.u_raw_holder.append(self.u_raw)
        self.length_list_holder.append(self.length_list)
        self.test_length_list_holder.append(self.test_length_list)

    def _transfer(self):
        self.x_raw = self.x_raw_holder
        self.x_test = self.x_test_holder
        self.u_test = self.u_test_holder
        self.u_raw = self.u_raw_holder
        self.length_list = self.length_list_holder
        self.test_length_list = self.test_length_list_holder

    def _generate_data(self, args):
        """Load data from environment
        Args:
            args: Various arguments and specifications
        """

        # Initialize array to hold states and actions
        x_original, u_original, length_list, total_steps = self._data_collection_loop(args, args['total_data_size'])

        # Generate test scenario
        x_test, u_test, test_length_list, test_steps = self._data_collection_loop(args, args['val_frac'] * args[
            'total_data_size'])
        self._store_internally(x_test, u_test, test_length_list, x_original, u_original, length_list, args)
        self.total_steps += total_steps

    def _data_collection_loop(self, args, data_size):
        x_original = []
        u_original = []
        total_steps = 0
        # Define progress bar
        bar = progressbar.ProgressBar(maxval=data_size).start()
        length_list = []
        done_list = []
        # Loop through episodes
        while True:
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
            u_trial = np.zeros((args['max_ep_steps'] - 1, args['act_dim']), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            for t in range(1, args['max_ep_steps']):
                action = self.env.action_space.sample()
                u_trial[t - 1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])

                if step_info[3]['data_collection_done']:
                    break

            total_steps += t - self.seq_length
            x_original.append(x_trial)
            u_original.append(u_trial)
            done_list.append(step_info[3]['data_collection_done'])
            length_list.append(t)
            if total_steps >= data_size:
                break
            bar.update(total_steps)
        bar.finish()

        return x_original, u_original, length_list, total_steps

    def _store_internally(self, x_test, u_test, test_length_list, x_original, u_original, length_list, args):

        """store and sort the collected data internally in memory"""

        if args['continue_data_collection']:

            ### Recover previously stored test data
            self.x_test = self.x_test * self.scale_x + self.shift_x
            self.u_test = self.u_test * self.scale_u + self.shift_u

            self.x_test = np.concatenate([self.x_test, x_test], axis=0)
            self.u_test = np.concatenate([self.u_test, u_test], axis=0)
            self.test_length_list = np.concatenate([self.test_length_list, np.array(test_length_list)], axis=0)

            self.x_raw = np.concatenate([self.x_raw, np.array(x_original)], axis=0)
            self.u_raw = np.concatenate([self.u_raw, np.array(u_original)], axis=0)
            self.length_list = np.concatenate([self.length_list, np.array(length_list)], axis=0)

        else:
            self.x_test = np.array(x_test)
            self.u_test = np.array(u_test)
            self.test_length_list = np.array(test_length_list)

            self.length_list = np.array(length_list)
            self.x_raw = np.array(x_original)
            self.u_raw = np.array(u_original)

    def _process_data(self, args):
        """Create batch dicts and shuffle data
        Args:
            args: Various arguments and specifications
        """

        x = []
        u = []
        for i in range(self.x_raw.shape[0]):
            j = 0
            while j + self.seq_length < self.length_list[i]:
                x.append(self.x_raw[i][j:j + self.seq_length])
                u.append(self.u_raw[i][j:j + self.seq_length - 1])
                j += 1
        x = np.array(x)
        u = np.array(u)
        # Reshape and trim data sets
        self.x = x.reshape(-1, self.seq_length, args['state_dim'])
        self.u = u.reshape(-1, self.seq_length - 1, args['act_dim'])
        len_x = int(np.floor(len(self.x) / args['batch_size']) * args['batch_size'])
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

        # Create batch_dict
        self.batch_dict = {}

        # Print tensor shapes
        print('states: ', self.x.shape)
        print('inputs: ', self.u.shape)

        self.batch_dict['states'] = np.zeros((args['batch_size'], self.seq_length, args['state_dim']))
        self.batch_dict['inputs'] = np.zeros((args['batch_size'], self.seq_length - 1, args['act_dim']))

        # Shuffle data before splitting into train/val
        print('shuffling...')
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.u = self.u[p]

        # prepare test set data

        x = []
        u = []
        for i in range(self.x_test.shape[0]):
            j = 0
            while j + self.seq_length < self.test_length_list[i]:
                x.append(self.x_test[i][j:j + self.seq_length])
                u.append(self.u_test[i][j:j + self.seq_length - 1])
                j += 1
        # j = 0
        # while j + self.seq_length < self.x_test.shape[0]:
        #     x.append(self.x_test[j:j + self.seq_length])
        #     u.append(self.u_test[j:j + self.seq_length - 1])
        #     j += 1
        x = np.array(x)
        u = np.array(u)
        self.x_test_batch = x.reshape(-1, self.seq_length, args['state_dim'])
        self.u_test_batch = u.reshape(-1, self.seq_length - 1, args['act_dim'])
        self.test_trajectory_index = np.random.randint(0, self.x_test.shape[0])

    def _process_data_meta(self, args):
        # Create batch_dict
        self.batch_dict = {'states': [], 'inputs': []}
        self.x_test_batch_list = []
        self.u_test_batch_list = []
        self.test_trajectory_index_list = []
        self.x = []
        self.u = []

        self.batch_dict['states'] = []
        self.batch_dict['inputs'] = []

        for i in range(self.task_num):
            x = []
            u = []
            for j in range(self.x_raw[i].shape[0]):
                k = 0
                while k + self.seq_length < self.length_list[i][j]:
                    x.append(self.x_raw[i][j][k:k + self.seq_length])
                    u.append(self.u_raw[i][j][k:k + self.seq_length - 1])
                    k += 1
            x = np.array(x)
            u = np.array(u)
            # Reshape and trim data sets
            self.x.append(x.reshape(-1, self.seq_length, args['state_dim']))
            self.u.append(u.reshape(-1, self.seq_length - 1, args['act_dim']))

            # the len_x and len_u could be divided by the batch
            len_x = int(np.floor(len(self.x[i]) / args['batch_size']) * args['batch_size'])
            self.x[i] = self.x[i][:len_x]
            self.u[i] = self.u[i][:len_x]

            # Print tensor shapes (might need to be move outside the loop once the task_num become massive
            print('states_task' + str(i) + ': ', self.x[i].shape)
            print('inputs_task' + str(i) + ': ', self.u[i].shape)

            # Shuffle data before splitting into train/val
            print('shuffling...')
            p = np.random.permutation(len(self.x[i]))
            self.x[i] = self.x[i][p]
            self.u[i] = self.u[i][p]

            # prepare test set data
            x = []
            u = []
            for j in range(self.x_test[i].shape[0]):
                k = 0
                while k + self.seq_length < self.test_length_list[i][j]:
                    x.append(self.x_test[i][j][k:k + self.seq_length])
                    u.append(self.u_test[i][j][k:k + self.seq_length - 1])
                    k += 1
            x = np.array(x)
            u = np.array(u)
            # only one batch for test set
            self.x_test_batch_list.append(x.reshape(-1, self.seq_length, args['state_dim']))
            self.u_test_batch_list.append(u.reshape(-1, self.seq_length - 1, args['act_dim']))
            self.test_trajectory_index_list.append(np.random.randint(0, self.x_test[i].shape[0]))
        print('h')

    def _create_split(self, args):
        """Divide data into training/validation sets
        Args:
            args: Various arguments and specifications
        """

        # Compute number of batches
        self.n_batches = len(self.x) // args['batch_size']
        self.n_batches_val = int(math.floor(args['val_frac'] * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        # Divide into train and validation datasets
        self.x_val = self.x[self.n_batches_train * args['batch_size']:]
        self.u_val = self.u[self.n_batches_train * args['batch_size']:]
        self.x = self.x[:self.n_batches_train * args['batch_size']]
        self.u = self.u[:self.n_batches_train * args['batch_size']]

        # Set batch pointer for training and validation sets
        self.reset_batchptr_train()
        self.reset_batchptr_val()

    def _create_split_meta(self, args):
        """Divide data into training/validation sets (list_version)
        Args:
            args: Various arguments and specifications
        """
        self.n_batches = []  # number of total batches per task, in integer
        self.n_batches_val = []  # number of validation batches per task, in integer
        self.n_batches_train = []  # number of train batches per task, in integer
        self.x_val = []  # validation set for states
        self.u_val = []

        for i in range(self.task_num):
            # Compute number of batches
            self.n_batches.append(len(self.x[i]) // args['batch_size'])
            self.n_batches_val.append(int(math.floor(args['val_frac'] * self.n_batches[i])))
            self.n_batches_train.append(self.n_batches[i] - self.n_batches_val[i])

            print('num training batches_task_' + str(i) + ': ', self.n_batches_train[i])
            print('num validation batches_task_' + str(i) + ': ', self.n_batches_val[i])

            # Divide into train and validation datasets
            self.x_val.append(self.x[i][self.n_batches_train[i] * args['batch_size']:])
            self.u_val.append(self.u[i][self.n_batches_train[i] * args['batch_size']:])
            self.x[i] = self.x[i][:self.n_batches_train[i] * args['batch_size']]
            self.u[i] = self.u[i][:self.n_batches_train[i] * args['batch_size']]

            # Set batch pointer for training and validation sets
        self.n_batches = np.asarray(self.n_batches)
        self.n_batches_val = np.asarray(self.n_batches_val)
        self.n_batches_train = np.asarray(self.n_batches_train)

    def _shift_scale(self, args):
        """Shift and scale data to be zero-mean, unit variance
        Args:
            args: Various arguments and specifications
        """
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            self._determine_shift_and_scale(args)

        # Shift and scale values for test sequence
        self.x_test = (self.x_test - self.shift_x) / self.scale_x
        self.u_test = (self.u_test - self.shift_u) / self.scale_u

    def _determine_shift_and_scale(self, args):
        self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1))
        self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1))
        self.shift_u = np.mean(self.u[:self.n_batches_train], axis=(0, 1))
        self.scale_u = np.std(self.u[:self.n_batches_train], axis=(0, 1))

        # Remove very small scale values
        self.scale_x[self.scale_x < 1e-6] = 1.0

    def _determine_shift_and_scale_meta(self, args):
        # determine shift and scale for actions and states, in list format
        self.shift_x = []
        self.scale_x = []
        self.shift_u = []
        self.scale_u = []
        for i in range(self.task_num):
            self.shift_x.append(np.mean(self.x[i][:self.n_batches_train[i]], axis=(0, 1)))
            self.scale_x.append(np.std(self.x[i][:self.n_batches_train[i]], axis=(0, 1)))
            self.shift_u.append(np.mean(self.u[i][:self.n_batches_train[i]], axis=(0, 1)))
            self.scale_u.append(np.std(self.u[i][:self.n_batches_train[i]], axis=(0, 1)))

            # Remove small scale values for x
            self.scale_x[i][self.scale_x[i] < 1e-6] = 1.0

        self.shift_x = np.asarray(self.shift_x)
        self.scale_x = np.asarray(self.scale_x)
        self.shift_u = np.asarray(self.shift_u)
        self.scale_u = np.asarray(self.scale_u)

    def update_data(self, x_new, u_new, val_frac):
        """Update training/validation data
        Args:
            x_new: New state values
            u_new: New control inputs
            val_frac: Fraction of new data to include in validation set
        """
        # First permute data
        p = np.random.permutation(len(x_new))
        x_new = x_new[p]
        u_new = u_new[p]

        # Divide new data into training and validation components
        n_seq_val = max(int(math.floor(val_frac * len(x_new))), 1)
        n_seq_train = len(x_new) - n_seq_val
        x_new_val = x_new[n_seq_train:]
        u_new_val = u_new[n_seq_train:]
        x_new = x_new[:n_seq_train]
        u_new = u_new[:n_seq_train]

        # Now update training and validation data
        self.x = np.concatenate((x_new, self.x), axis=0)
        self.u = np.concatenate((u_new, self.u), axis=0)
        self.x_val = np.concatenate((x_new_val, self.x_val), axis=0)
        self.u_val = np.concatenate((u_new_val, self.u_val), axis=0)

        # Update sizes of train and val sets
        self.n_batches_train = len(self.x) // self.batch_size
        self.n_batches_val = len(self.x_val) // self.batch_size

    def next_batch_train(self):
        """Sample a new batch from training data
        Args:
            None
        Returns:
            batch_dict: Batch of training data
        """
        # Extract next batch

        batch_index = self.batch_permuation_train[
                      self.batchptr_train * self.batch_size:(self.batchptr_train + 1) * self.batch_size]
        self.batch_dict['states'] = (self.x[batch_index] - self.shift_x) / self.scale_x
        self.batch_dict['inputs'] = (self.u[batch_index] - self.shift_u) / self.scale_u

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    def next_batch_train_meta(self):

        # shuffle x and u every time
        self.batch_dict['states'] = []
        self.batch_dict['inputs'] = []

        x = []
        u = []

        # create random array to pick up items from x and u randomly
        for i in range(self.task_num):
            p = np.random.permutation(len(self.x[i]))[0:self.batch_size]
            x.append(self.x[i][p])
            u.append(self.u[i][p])

        for i in range(self.task_num):
            self.batch_dict['states'].append((x[i] - self.shift_x[i]) / self.scale_x[i])
            self.batch_dict['inputs'].append((u[i] - self.shift_u[i]) / self.scale_u[i])

        return self.batch_dict

    def random_sample(self):

        batch_index = np.random.choice(self.x.shape[0],
                                       size=self.batch_size, replace=False)

        self.batch_dict['states'] = (self.x[batch_index] - self.shift_x) / self.scale_x
        self.batch_dict['inputs'] = (self.u[batch_index] - self.shift_u) / self.scale_u

        return self.batch_dict

    def reset_batchptr_train(self):
        """Reset pointer to first batch in training set
        Args:
            None
        """
        self.batch_permuation_train = np.random.permutation(len(self.x))
        self.batchptr_train = 0

    def next_batch_val(self):
        """Sample a new batch from validation data
        Args:
            None
        Returns:
            batch_dict: Batch of validation data
        """
        # Extract next validation batch

        batch_index = range(self.batchptr_val * self.batch_size, (self.batchptr_val + 1) * self.batch_size)
        self.batch_dict['states'] = (self.x_val[batch_index] - self.shift_x) / self.scale_x
        self.batch_dict['inputs'] = (self.u_val[batch_index] - self.shift_u) / self.scale_u

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    def get_all_val_data(self):
        self.batch_dict['states'] = []
        self.batch_dict['inputs'] = []

        min_num = np.amin([i.shape[0] for i in self.x_val])

        x = []
        u = []

        # create random array to pick up items from x and u randomly
        for i in range(self.task_num):
            p = np.random.permutation(len(self.x_val[i]))[0:min_num]
            x.append(self.x_val[i][p])
            u.append(self.u_val[i][p])

        for i in range(self.task_num):
            self.batch_dict['states'].append((x[i] - self.shift_x[i]) / self.scale_x[i])
            self.batch_dict['inputs'].append((u[i] - self.shift_u[i]) / self.scale_u[i])
        return self.batch_dict

    def get_all_test_data(self):
        self.batch_dict['states'] = []
        self.batch_dict['inputs'] = []

        min_num = np.amin([i.shape[0] for i in self.x_test_batch_list])

        x = []
        u = []

        # create random array to pick up items from x and u randomly
        for i in range(self.task_num):
            p = np.random.permutation(len(self.x_test_batch_list[i]))[0:min_num]
            x.append(self.x_test_batch_list[i][p])
            u.append(self.u_test_batch_list[i][p])

        for i in range(self.task_num):
            self.batch_dict['states'].append(x[i])
            self.batch_dict['inputs'].append(u[i])
        return self.batch_dict

    def get_all_train_data(self):
        # Extract next batch

        self.batch_dict['states'] = []
        self.batch_dict['inputs'] = []

        min_num = np.amin([i.shape[0] for i in self.x])

        x = []
        u = []

        # create random array to pick up items from x and u randomly
        for i in range(self.task_num):
            p = np.random.permutation(len(self.x[i]))[0:min_num]
            x.append(self.x[i][p])
            u.append(self.u[i][p])

        for i in range(self.task_num):
            self.batch_dict['states'].append((x[i] - self.shift_x[i]) / self.scale_x[i])
            self.batch_dict['inputs'].append((u[i] - self.shift_u[i]) / self.scale_u[i])
        return self.batch_dict

    def get_test_trajectory(self):
        return self.x_test[self.test_trajectory_index], self.u_test[self.test_trajectory_index]

    def get_test_trajectory_meta(self):
        a = []
        b = []
        for i in range(self.task_num):
            a.append(self.x_test[i][self.test_trajectory_index_list[i]])
            b.append(self.u_test[i][self.test_trajectory_index_list[i]])
        return a, b

    def save_data(self, path):
        os.makedirs(path, exist_ok=True)
        counter = 0

        for entry in os.scandir(path):
            if entry.path.endswith(".npy"):
                counter += 1
            if entry.path.endswith("0.npy"):  # might be useless
                counter -= 1

        counter = int(counter / 6)  # indicate how many tasks we have

        # name task from 1, 0 is special for merged quest
        np.save(path + '/x_' + str(counter + 1) + '.npy', self.x_raw)
        np.save(path + '/u_' + str(counter + 1) + '.npy', self.u_raw)
        np.save(path + '/x_test_' + str(counter + 1) + '.npy', self.x_test)
        np.save(path + '/u_test_' + str(counter + 1) + '.npy', self.u_test)

        np.save(path + '/length_list_' + str(counter + 1) + '.npy', self.length_list)
        np.save(path + '/test_length_list_' + str(counter + 1) + '.npy', self.test_length_list)

    def _restore_data(self, task_num, path):
        # only go through the given task
        self.x_raw = np.load(path + '/x_' + str(task_num) + '.npy')
        self.u_raw = np.load(path + '/u_' + str(task_num) + '.npy')
        self.x_test = np.load(path + '/x_test_' + str(task_num) + '.npy')
        self.u_test = np.load(path + '/u_test_' + str(task_num) + '.npy')

        self.length_list = np.load(path + '/length_list_' + str(task_num) + '.npy')
        self.test_length_list = np.load(path + '/test_length_list_' + str(task_num) + '.npy')

    def _restore_data_meta(self, path):
        self.x_raw = []
        self.u_raw = []
        self.x_test = []
        self.u_test = []

        self.length_list = []
        self.test_length_list = []

        # go through the files in the folder, start from 1 end in task_num
        for i in range(1, self.task_num + 1):
            self.x_raw.append(np.load(path + '/x_' + str(i) + '.npy'))
            self.u_raw.append(np.load(path + '/u_' + str(i) + '.npy'))
            self.x_test.append(np.load(path + '/x_test_' + str(i) + '.npy'))
            self.u_test.append(np.load(path + '/u_test_' + str(i) + '.npy'))

            self.length_list.append(np.load(path + '/length_list_' + str(i) + '.npy'))
            self.test_length_list.append(np.load(path + '/test_length_list_' + str(i) + '.npy'))

    def _merging_data(self):
        # add combined data to different lists at the first dimension
        x_0 = np.concatenate(self.x_raw, axis=0)
        u_0 = np.concatenate(self.u_raw, axis=0)
        x_test_0 = np.concatenate(self.x_test, axis=0)
        u_test_0 = np.concatenate(self.u_test, axis=0)

        length_list_0 = np.concatenate(self.length_list, axis=0)
        test_length_list_0 = np.concatenate(self.test_length_list, axis=0)

        self.x_raw.insert(0, x_0)
        self.u_raw.insert(0, u_0)
        self.x_test.insert(0, x_test_0)
        self.u_test.insert(0, u_test_0)

        self.length_list.insert(0, length_list_0)
        self.test_length_list.insert(0, test_length_list_0)

        self.task_num = self.task_num + 1

    def _generate_data_with_controller(self, controller, args):
        # Initialize array to hold states and actions
        x = []
        u = []

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args['total_data_size']).start()

        # Loop through episodes
        while True:
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
            u_trial = np.zeros((args['max_ep_steps'] - 1, args['act_dim']), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            controller.update_reference(self.env.reference)
            for t in range(1, args['max_ep_steps']):
                action = controller.choose_action(x_trial[t - 1], self.env.reference)
                u_trial[t - 1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])
                if step_info[3]['data_collection_done']:
                    break
            j = 0
            while j + self.seq_length < len(x_trial):
                x.append(x_trial[j:j + self.seq_length])
                u.append(u_trial[j:j + self.seq_length - 1])
                j += 1

            if len(x) >= args['total_data_size']:
                break
            bar.update(len(x))
        bar.finish()

        # Generate test scenario
        x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
        u_trial = np.zeros((args['max_ep_steps'] - 1, args['act_dim']), dtype=np.float32)
        # Reset environment and simulate with random actions
        x_trial[0] = self.env.reset()
        controller.update_reference(self.env.reference)
        for t in range(1, args['max_ep_steps']):
            action = controller.choose_action(x_trial[t - 1], self.env.reference)
            u_trial[t - 1] = action
            step_info = self.env.step(action)
            x_trial[t] = np.squeeze(step_info[0])
            if step_info[3]['data_collection_done']:
                break

        self.x_test = x_trial
        self.u_test = u_trial

        x = np.array(x)
        u = np.array(u)
        # Reshape and trim data sets
        x = x.reshape(-1, self.seq_length, args['state_dim'])
        u = u.reshape(-1, self.seq_length - 1, args['act_dim'])
        self.x = np.concatenate([self.x, x], axis=0)
        self.u = np.concatenate([self.u, u], axis=0)
        len_x = int(np.floor(len(self.x) / args['batch_size']) * args['batch_size'])
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

    def update_and_process_data_with_controller(self, controller, args):

        self._generate_data_with_controller(controller, args)

        self._process_data(args)

        print('creating splits...')
        self._create_split(args)

        print('shifting/scaling data...')
        self._shift_scale(args)

    def reset_batchptr_val(self):
        """Reset pointer to first batch in validation set
        Args:
            None
        """
        self.batchptr_val = 0

