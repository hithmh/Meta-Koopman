


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy

class three_tank_system(gym.Env):

    def __init__(self, ref = np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877])):
        self.t = 0
        self.real_world_time = 0  # second

        self.action_sample_period = 20
        self.sampling_period = 0.005
        self.h = 0.001
        self.sampling_steps = int(self.sampling_period/self.h)
        self.delay = 5
        self.delay_dims = [0, 1,  3, 4,  6, 7]

        self.s2hr = 3600
        self.MW = 250e-3
        self.sum_c = 2E3
        self.T10 = 300
        self.T20 = 300
        self.F10 = 5.04
        self.F20 = 5.04
        self.Fr = 50.4
        self.Fp = 0.504
        self.V1 = 1
        self.V2 = 0.5
        self.V3 = 1
        self.E1 = 5e4
        self.E2 = 6e4
        self.k1 = 2.77e3 * self.s2hr
        self.k2 = 2.6e3 * self.s2hr
        self.dH1 = -6e4 / self.MW
        self.dH2 = -7e4 / self.MW
        self.aA = 3.5
        self.aB = 1
        self.aC = 0.5
        self.Cp = 4.2e3
        self.R = 8.314
        self.rho = 1000
        self.xA10 = 1
        self.xB10 = 0
        self.xA20 = 1
        self.xB20 = 0
        self.Hvap1 = -35.3E3 * self.sum_c
        self.Hvap2 = -15.7E3 * self.sum_c
        self.Hvap3 = -40.68E3 * self.sum_c

        self.kw = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # noise deviation
        self.bw = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5]) # noise bound

        self.xs = np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877])
        self.us = 1.12 * np.array([2.9e9, 1.0e9, 2.9e9])

        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.action_low = 0.2 * self.us
        self.action_high = 1.5 * self.us

        self.reference = ref

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.state_buffer = state_buffer(self.delay, self.delay_dims)

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action, impulse = 0):

        action = np.clip(action, self.action_low, self.action_high)
        
        x0 = self.state
        for i in range(self.sampling_steps):
            process_noise = np.random.normal(np.zeros_like(self.kw),self.kw)
            process_noise = np.clip(process_noise, -self.bw, self.bw)
            x0 = x0 + self.derivative(x0, action)*self.h + process_noise*self.h
            
        self.state = x0
        self.state_buffer.memorize(x0)
        self.t += 1
        s = self.state_buffer.get_state(self.t)

        self.time = self.t * self.sampling_period
        
        cost = np.linalg.norm(self.state - self.reference)
        done = False
        data_collection_done = False
        
        return s, cost, done, dict(reference=self.reference, data_collection_done=data_collection_done)

    def reset(self):

        self.state_buffer.reset()
        self.t = 0
        self.time = 0
        self.a_holder = self.action_space.sample()
        self.state = np.random.uniform(0.8, 1.2) * self.xs + np.random.normal(np.zeros_like(self.xs), self.xs*0.01)

        while True:
            self.state_buffer.memorize(self.state)
            self.t += 1
            s = self.state_buffer.get_state(self.t)
            if s is not None:
                break

            action = self.action_space.sample()
            for i in range(self.sampling_steps):
                process_noise = np.random.normal(np.zeros_like(self.kw), self.kw)
                process_noise = np.clip(process_noise, -self.bw, self.bw)
                self.state = self.state + self.derivative(self.state, action) * self.h + process_noise * self.h
        self.time = self.t * self.sampling_period
        return s

    
    def derivative(self, x, us):
        xA1 = x[0]
        xB1 = x[1]
        T1 = x[2]

        xA2 = x[3]
        xB2 = x[4]
        T2 = x[5]

        xA3 = x[6]
        xB3 = x[7]
        T3 = x[8]

        Q1 = us[0]
        Q2 = us[1]
        Q3 = us[2]

        xC3 = 1 - xA3 - xB3
        x3a = self.aA * xA3 + self.aB * xB3 + self.aC * xC3

        xAr = self.aA * xA3 / x3a
        xBr = self.aB * xB3 / x3a
        xCr = self.aC * xC3 / x3a

        F1 = self.F10 + self.Fr
        F2 = F1 + self.F20
        F3 = F2 - self.Fr - self.Fp

        f1 = self.F10 * (self.xA10 - xA1) / self.V1 + self.Fr * (xAr - xA1) / self.V1 - self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
        f2 = self.F10 * (self.xB10 - xB1) / self.V1 + self.Fr * (xBr - xB1) / self.V1 + self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1 - self.k2 * np.exp(
            -self.E2 / (self.R * T1)) * xB1
        f3 = self.F10 * (self.T10 - T1) / self.V1 + self.Fr * (T3 - T1) / self.V1 - self.dH1 * self.k1 * np.exp(
            -self.E1 / (self.R * T1)) * xA1 / self.Cp - self.dH2 * self.k2 * np.exp(
            -self.E2 / (self.R * T1)) * xB1 / self.Cp + Q1 / (self.rho * self.Cp * self.V1)

        f4 = F1 * (xA1 - xA2) / self.V2 + self.F20 * (self.xA20 - xA2) / self.V2 - self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2
        f5 = F1 * (xB1 - xB2) / self.V2 + self.F20 * (self.xB20 - xB2) / self.V2 + self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2 - self.k2 * np.exp(
            -self.E2 / (self.R * T2)) * xB2
        f6 = F1 * (T1 - T2) / self.V2 + self.F20 * (self.T20 - T2) / self.V2 - self.dH1 * self.k1 * np.exp(
            -self.E1 / (self.R * T2)) * xA2 / self.Cp - self.dH2 * self.k2 * np.exp(
            -self.E2 / (self.R * T2)) * xB2 / self.Cp + Q2 / (self.rho * self.Cp * self.V2)

        f7 = F2 * (xA2 - xA3) / self.V3 - (self.Fr + self.Fp) * (xAr - xA3) / self.V3
        f8 = F2 * (xB2 - xB3) / self.V3 - (self.Fr + self.Fp) * (xBr - xB3) / self.V3
        f9 = F2 * (T2 - T3) /self.V3 + Q3 / (self.rho * self.Cp * self.V3) + (self.Fr + self.Fp) * (xAr * self.Hvap1 + xBr * self.Hvap2 + xCr * self.Hvap3) / (
                self.rho * self.Cp * self.V3)

        F = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])

        return F
    
    def render(self, mode='human'):

        return


    def get_action(self):

        if self.t % self.action_sample_period == 0:
            self.a_holder = self.action_space.sample()
        a = self.a_holder + np.random.normal(np.zeros_like(self.us), self.us*0.01)
        a = np.clip(a, self.action_low, self.action_high)

        return a

class state_buffer(object):

    def __init__(self, delay, delay_dims):

        self.delay = delay
        self.delay_dims = delay_dims
        self.memory = []


    def memorize(self, s):
        s = copy.copy(s)
        self.memory.append(s)
        return

    def get_state(self, t):

        if t < self.delay:
            return None
        else:
            s = copy.copy(self.memory[-1])
            # for i in self.delay_dims:
            #     s[i] = self.memory[t-self.delay][i]
            s[self.delay_dims] = self.memory[t - self.delay][self.delay_dims]
            return s

    def reset(self):
        self.memory = []

if __name__=='__main__':
    env = three_tank_system()
    T = 20000
    path = []
    a_path = []
    t1 = []
    s = env.reset()
    for i in range(int(T)):
        action = env.us
        s, r, done, info = env.step(action)
        path.append(s)
        a_path.append(action)
        t1.append(i)
    path = np.array(path)
    state_dim = s.shape[0]
    fig, ax = plt.subplots(state_dim, sharex=True, figsize=(15, 15))
    t = range(T)
    for i in range(state_dim):
        ax[i].plot(t, path[:, i], color='red')
    # fig = plt.figure(figsize=(9, 6))
    # ax = fig.add_subplot(111)
    # ax.plot(t1, path)
    # # ax.plot(t2, path2, color='red',label='1')
    # #
    # # ax.plot(t3, path3, color='black', label='0.01')
    # # ax.plot(t4, path4, color='orange', label='0.001')
    # handles, labels = ax.get_legend_handles_labels()
    #
    # ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    # plt.show()
    # print('done')

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, a_path)
    # ax.plot(t2, path2, color='red',label='1')
    #
    # ax.plot(t3, path3, color='black', label='0.01')
    # ax.plot(t4, path4, color='orange', label='0.001')
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')









