import numpy as np
import sys
import io
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

    def __init__(self, shape=(5, 6), num_ckpt=3, num_blocks=3):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = shape[0] * shape[1]
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        # special states
        # SS = np.random.choice(np.arange(nS), num_ckpt+num_blocks+2, replace=False)
        # self.start = SS[0]
        # self.goal = SS[1]
        # self.blocks = list(SS[2:num_blocks+2])
        # self.checkpoints = list(SS[num_blocks+2:])
        self.start = 20
        self.goal = 7
        self.blocks = [12, 13, 14]
        self.checkpoints = [23]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == self.goal
            if is_done(s):
                reward = 0.0
            elif s in self.checkpoints:
                reward = -0.1
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

        # We always start in generated start state
        isd = np.zeros(nS)
        isd[self.start] = 1.0

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(TownEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " S "
            elif s == self.goal:
                output = " G "
            elif s in self.checkpoints:
                output = " X "
            elif s in self.blocks:
                output = " # "
            else:
                output = " . "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
