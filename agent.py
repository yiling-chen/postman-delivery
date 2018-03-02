import sys
import argparse
import numpy as np
from townworld import TownEnv
from collections import defaultdict


def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    '''
    Update the Q value with a fraction (defined by the learning rate, alpha)
    of TD target.
    '''
    return Qsa + alpha * (reward + gamma * Qsa_next - Qsa)


def q_from_v(env, V, s, gamma=1.0):
    '''
    Bellman Expecation equation for the action-value function Q(s, a)
    Note that here we evaluate all action values of a specified state s.
    '''
    q = np.zeros(env.nA)

    for a in range(env.nA):
        for prob, next_s, r, done in env.P[s][a]:
            q[a] += prob * (r + gamma * V[next_s])

    return q


def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
    '''
    return a greedy policy that chooses the action with largest action values (exploitation)
           with a probibility of epsilon of random exploration.
    epsilon: the rate of exploration, decays with the number of episode
    '''
    epsilon = 1.0 / i_episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
    return policy_s


class Agent:
    def __init__(self, env, alpha, gamma=1.0):
        self.env = env
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma

    def value_iteration(self, theta=1e-8, msg=None):
        msg = "Exploring nearest unvisited checkpoint" if msg is None else msg

        iter = 1
        V = np.zeros(self.env.nS)
        while True:
            print('\r{}... iteration {}'.format(msg, iter), end="")
            sys.stdout.flush()
            iter += 1

            delta = 0
            for s in range(self.env.nS):
                v = V[s]
                q_s = q_from_v(self.env, V, s, self.gamma)
                V[s] = np.amax(q_s)
                delta = max(delta, abs(V[s] - v))
            if delta < theta:
                break

        self.Q = defaultdict(lambda: np.zeros(env.nA))
        for s in range(self.env.nS):
            self.Q[s] = q_from_v(self.env, V, s, self.gamma)

        sys.stdout.write('\n')
        return self.Q

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

    def explore(self, use_dynamic_programming=True, num_episodes=5000, msg=None):
        if use_dynamic_programming:
            return self.value_iteration(msg=msg)
        else:
            return self.q_learning(num_episodes, msg)

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
            if dist >= self.env.shape[0] + self.env.shape[1]:
                raise ValueError("Detected a loop! The learning process may not converge. "
                                 "Consider to increase episode number.")

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


'''
This is a variant of Traveling Salesman Problem (TSP).
This program implements the heuristic solution of nearest neighbor search strategy.
https://en.wikipedia.org/wiki/Travelling_salesman_problem#Constructive_heuristics
Briefly, the postman chooses the nearest unvisited target (checkpoints or final goal)
as the next move. For this sub-problem, the shortest path can be efficiently solved
by dynamic programming. This requires a full knowledge of the environment.
Alternatively, we can solve this problem by a model-free approach. This program implements
a temporal difference learning algorithm (Q-learning) to let an agent learn the optimal 
policy to reach the goal by exploring the grid environment. 
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solving postman delivery problem by using dynamic programming or '
                    'temporal difference learning (Q-learning)\n'
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
    parser.add_argument(
        '-d', '--dynamic_programming', type=int, default=1,
        help="Use dynamic programming. [default=1 (Set to 0 to disable)]."
    )

    args = parser.parse_args()

    height = args.height
    width = args.width
    num_ckpt = args.checkpoint
    num_blocks = args.blocks
    num_episodes = args.episode
    use_DP = args.dynamic_programming

    # Special states - randomly picks some locations as start, target and checkpoints etc.
    SS = np.random.choice(np.arange(height*width), num_ckpt+num_blocks+2, replace=False)
    start = SS[0]
    goal = SS[1]
    blocks = list(SS[2:num_blocks+2])
    checkpoints = list(SS[num_blocks+2:])

    total_travel_distance = 0

    env = TownEnv(start, goal, blocks, shape=(height, width))
    # plot the whole environment configuration
    print("Environment configuration:\nS(start), G(goal), X(checkpoint), #(obstacles)")
    env.plot(start, goal, checkpoints)

    # create an agent to explore the environment
    agent = Agent(env, alpha=0.01)

    last_start = start

    if len(checkpoints) > 0:
        while len(checkpoints) > 1:
            Q_matrices = []
            distances = np.zeros(len(checkpoints))
            for i in range(len(checkpoints)):
                env.reset_env_dynamics(last_start, checkpoints[i])
                Q_matrices.append(agent.explore(use_DP, num_episodes))
                distances[i] = agent.extract_shortest_path(Q_matrices[-1], last_start, checkpoints[i])

            idx = np.argmin(distances)
            total_travel_distance += agent.extract_shortest_path(Q_matrices[idx], last_start, checkpoints[idx], True)
            last_start = checkpoints[idx]
            del checkpoints[idx]

        # last checkpoint
        env.reset_env_dynamics(last_start, checkpoints[0])
        Q = agent.explore(use_DP, num_episodes)
        total_travel_distance += agent.extract_shortest_path(Q, last_start, checkpoints[0], True)
        last_start = checkpoints[0]

    env.reset_env_dynamics(last_start, goal)
    Q = agent.explore(use_DP, num_episodes, "Reaching the goal")
    total_travel_distance += agent.extract_shortest_path(Q, last_start, goal, True)
    print("Total travel distance:", total_travel_distance)
