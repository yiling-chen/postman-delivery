import sys
import gym
import pylab
import numpy as np
from townworld import TownEnv
from collections import defaultdict

def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    return Qsa + alpha * (reward + gamma * Qsa_next - Qsa)

# def epsilon_greedy_probs(env, Q_s, epsilon):
#     policy_s = np.ones(env.nA) * epsilon / env.nA
#     policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
#     return policy_s

def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
    epsilon = 1.0 / i_episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
    return policy_s

class TDAgent:
    def __init__(self, env, alpha, gamma=1.0):
        self.env = env
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    def q_learning(self, num_episodes):
        self.Q = defaultdict(lambda: np.zeros(env.nA))
        for i in range(1, num_episodes+1):
            if i % 100 == 0:
                print('\rEpisode {}/{}'.format(i, num_episodes), end="")
                sys.stdout.flush()
            state = self.env.reset()
            while True:
                # if self.epsilon > self.epsilon_min:
                #     self.epsilon *= self.epsilon_decay
                # policy_s = epsilon_greedy_probs(self.env, self.Q[state], self.epsilon)
                policy_s = epsilon_greedy_probs(self.env, self.Q[state], i)
                action = np.random.choice(np.arange(self.env.nA), p=policy_s)
                next_state, reward, done, info = self.env.step(action)
                self.Q[state][action] = update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward, self.alpha, self.gamma)
                state = next_state
                if done:
                    break

    def print_policy(self):
        print()
        policy_sarsamax = np.array([np.argmax(self.Q[key]) if key in self.Q else -1 for key in np.arange(self.env.nS)]).reshape(self.env.shape)
        # print(policy_sarsamax)
        grid = np.arange(self.env.nS).reshape(self.env.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if policy_sarsamax[y][x] == 0:
                output = " ↑ "
            elif policy_sarsamax[y][x] == 1:
                output = " → "
            elif policy_sarsamax[y][x] == 2:
                output = " ↓ "
            elif policy_sarsamax[y][x] == 3:
                output = " ← "
            else:
                output = " . "

            if x == 0:
                output = output.lstrip()
            if x == self.env.shape[1] - 1:
                output = output.rstrip()

            sys.stdout.write(output)

            if x == self.env.shape[1] - 1:
                sys.stdout.write("\n")

            it.iternext()

if __name__ == "__main__":
    # env = gym.make('CliffWalking-v0')
    env = TownEnv(shape=(4, 6), num_ckpt=1)
    agent = TDAgent(env, alpha=0.01)

    # initial environment configuration
    env._render()
    agent.q_learning(5000)
    agent.print_policy()
