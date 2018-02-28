import sys
import argparse
import numpy as np
from townworld import TownEnv
from collections import defaultdict


def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    return Qsa + alpha * (reward + gamma * Qsa_next - Qsa)


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

    def q_learning(self, num_episodes, msg=None):
        msg = "Exploring nearest unvisited checkpoint" if msg is None else msg
        self.Q = defaultdict(lambda: np.zeros(env.nA))
        for i in range(1, num_episodes+1):
            if i % 100 == 0:
                print('\r{}... episode {}/{}'.format(msg, i, num_episodes), end="")
                sys.stdout.flush()
            state = self.env.reset()
            while True:
                policy_s = epsilon_greedy_probs(self.env, self.Q[state], i)
                action = np.random.choice(np.arange(self.env.nA), p=policy_s)
                next_state, reward, done, info = self.env.step(action)
                self.Q[state][action] = update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward, self.alpha, self.gamma)
                state = next_state
                if done:
                    break

        sys.stdout.write('\n')
        return self.Q

    def print_policy(self):
        # print()
        policy_sarsamax = np.array([np.argmax(self.Q[key]) if key in self.Q else -1 for key in np.arange(self.env.nS)]).reshape(self.env.shape)
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
                output = " # "

            if x == 0:
                output = output.lstrip()
            if x == self.env.shape[1] - 1:
                output = output.rstrip()

            sys.stdout.write(output)

            if x == self.env.shape[1] - 1:
                sys.stdout.write("\n")

            it.iternext()

        print()

    def extract_shortest_path(self, Q, start, goal, plot=False):
        policy_sarsamax = np.array(
            [np.argmax(Q[key]) if key in Q else -1 for key in np.arange(self.env.nS)])

        # trace route
        MAX_X = self.env.shape[1]
        grid = np.zeros(self.env.nS)    # flag of route
        next_state = start
        dist = 0
        while next_state != goal:
            grid[next_state] = 1
            dist += 1
            if policy_sarsamax[next_state] == 0:
                next_state -= MAX_X
            elif policy_sarsamax[next_state] == 1:
                next_state += 1
            elif policy_sarsamax[next_state] == 2:
                next_state += MAX_X
            elif policy_sarsamax[next_state] == 3:
                next_state -= 1

        if not plot:
            return dist

        # render shortest path
        policy_sarsamax = policy_sarsamax.reshape(self.env.shape)
        grid = grid.reshape(self.env.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if grid[y][x] > 0 and policy_sarsamax[y][x] == 0:
                output = " ↑ "
            elif grid[y][x] > 0 and policy_sarsamax[y][x] == 1:
                output = " → "
            elif grid[y][x] > 0 and policy_sarsamax[y][x] == 2:
                output = " ↓ "
            elif grid[y][x] > 0 and policy_sarsamax[y][x] == 3:
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

        print("Distance: {}".format(dist))
        sys.stdout.write("\n")
        return dist


if __name__ == "__main__":
    # add arg parser
    parser = argparse.ArgumentParser(
        description='Solving postman delivery problem by using temporal difference learning (Q-learning)\n'
                    'See for comments for detailed problem descriptions.\n')
    parser.add_argument(
        '-x', '--width', type=int, default=8,
        help="Width of the town environment. [default=8]."
    )
    parser.add_argument(
        '-y', '--height', type=int, default=5,
        help="Width of the town environment. [default=5]."
    )
    parser.add_argument(
        '-c', '--checkpoint', type=int, default=3,
        help="Number of checkpoints. [default=3]."
    )
    parser.add_argument(
        '-b', '--blocks', type=int, default=5,
        help="Number of blocks. [default=5]."
    )
    parser.add_argument(
        '-e', '--episode', type=int, default=5000,
        help="Number of episodes for Q-learning. [default=5000]."
    )

    args = parser.parse_args()

    height = args.height
    width = args.width
    num_ckpt = args.checkpoint
    num_blocks = args.blocks
    num_episodes = args.episode

    # special states
    SS = np.random.choice(np.arange(height*width), num_ckpt+num_blocks+2, replace=False)
    start = SS[0]
    goal = SS[1]
    blocks = list(SS[2:num_blocks+2])
    checkpoints = list(SS[num_blocks+2:])

    # start = 20
    # goal = 7
    # checkpoints = [24, 5]
    # blocks = [12, 13, 14]
    total_travel_distance = 0

    env = TownEnv(start, goal, blocks, shape=(height, width))
    # plot the whole environment configuration
    print("Environment configuration:\nS(start), G(goal), X(checkpoint), #(obstacles)")
    env.plot(start, goal, checkpoints)

    # create an agent to explore the environment
    agent = TDAgent(env, alpha=0.01)

    last_start = start

    if len(checkpoints) > 0:
        while len(checkpoints) > 1:
            Q_matrices = []
            distances = np.zeros(len(checkpoints))
            for i in range(len(checkpoints)):
                env.reset_env_dynamics(last_start, checkpoints[i])
                Q_matrices.append(agent.q_learning(num_episodes))
                distances[i] = agent.extract_shortest_path(Q_matrices[-1], last_start, checkpoints[i])

            idx = np.argmin(distances)
            total_travel_distance += agent.extract_shortest_path(Q_matrices[idx], last_start, checkpoints[idx], True)
            last_start = checkpoints[idx]
            del checkpoints[idx]

        # last checkpoint
        env.reset_env_dynamics(last_start, checkpoints[0])
        Q = agent.q_learning(num_episodes)
        total_travel_distance += agent.extract_shortest_path(Q, last_start, checkpoints[0], True)
        last_start = checkpoints[0]

    env.reset_env_dynamics(last_start, goal)
    # env.plot(None, goal, checkpoints)
    Q = agent.q_learning(num_episodes, "Reaching the goal")
    total_travel_distance += agent.extract_shortest_path(Q, last_start, goal, True)
    print("Total travel distance:", total_travel_distance)
