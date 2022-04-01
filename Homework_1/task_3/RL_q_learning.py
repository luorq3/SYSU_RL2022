
import numpy as np
import pandas as pd


class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        ''' build q table'''
        ############################

        self.q_table = pd.DataFrame(columns=self.actions)

        ############################

    def choose_action(self, observation):
        ''' choose action from q table '''
        ############################

        series = pd.Series(self.q_table.loc[observation])

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = series.argmax()
            action = series.index[action]

        return action

        ############################

    def learn(self, s, a, r, s_):
        ''' update q table '''
        ############################

        self.check_state_exist(s_)
        q_s = self.q_table.loc[s, a]
        q_s_ = self.q_table.loc[s_, :].max()
        td_error = q_s - r - self.gamma * q_s_
        q_ = q_s - self.lr * td_error
        self.q_table.loc[s, a] = q_

        ############################

    def check_state_exist(self, state):
        ''' check state '''
        ############################

        if state not in self.q_table.index:
            self.q_table.loc[state] = pd.Series(np.zeros(len(self.actions)), index=self.actions)

        ############################