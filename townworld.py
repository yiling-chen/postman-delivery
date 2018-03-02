import sys
import numpy as np
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class TownEnv(discrete.DiscreteEnv):
    """
    Town environment for postman shortest path problem.
    You are an postman in a town represented as a MxN grid.
    Starting from S and your goal is to reach the terminal G.
    On your way, you have to deliver mails at some checkpoints (X).
    Find the optimal policy to travel from S to G while passing all X.

    For example, a 4x6 grid looks as follows:

    .  .  .  .  .  .
    .  G  .  .  .  .
    #  #  #  .  .  .
    .  .  S  .  .  X

    Note: '#' are closed-blocks that the postman cannot pass.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, start, goal, blocks, shape=(5, 6)):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        # environment parameters
        self.shape = shape
        self.nS = shape[0] * shape[1]
        self.nA = 4

        self.start = start
        self.goal = goal
        self.blocks = blocks
        self.P = None

        self.reset_env_dynamics(self.start, self.goal)

    def reset_env_dynamics(self, start, goal):
        self.start = start
        self.goal = goal

        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]

        P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(self.nA)}

            is_done = lambda s: s == goal
            if is_done(s):
                reward = 0.0
            else:
                reward = -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if (y == 0 or (s - MAX_X) in self.blocks) else s - MAX_X
                ns_right = s if (x == (MAX_X - 1) or (s + 1) in self.blocks) else s + 1
                ns_down = s if (y == (MAX_Y - 1) or (s + MAX_X) in self.blocks) else s + MAX_X
                ns_left = s if (x == 0 or (s - 1) in self.blocks) else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Expose the underlying dynamics of the environment
        # Only for model-based method (DP)
        self.P = P

        # We always start in generated start state
        isd = np.zeros(self.nS)
        isd[start] = 1.0

        super(TownEnv, self).__init__(self.nS, self.nA, P, isd)

    def plot(self, start, goal, checkpoints):
        """
        Render the environment configuration
        """
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if start is not None and s == start:
                output = " S "
            elif goal is not None and s == goal:
                output = " G "
            elif checkpoints is not None and s in checkpoints:
                output = " X "
            elif s in self.blocks:
                output = " # "
            else:
                output = " . "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            sys.stdout.write(output)

            if x == self.shape[1] - 1:
                sys.stdout.write("\n")

            it.iternext()

        print()
