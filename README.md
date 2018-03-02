# Postman delivery shortest path

Town environment for postman shortest path problem.
You are an postman in a town represented as a MxN grid.
Starting from S and your goal is to reach the terminal G.
On your way, you have to deliver mails at some checkpoints (X).
Find the optimal policy to travel from S to G while passing all X.

For example, a possible 4x6 grid looks as follows:
```
.  .  .  .  .  .
.  G  .  .  .  .
#  #  #  .  .  .
.  .  S  .  .  X
```
Note: '#' are closed-blocks that the postman cannot pass.

You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
Actions going off the edge leave you in your current state.
You receive a reward of -1 at each step until you reach a terminal state.

# Dependency
You will need OpenAI `gym` and `numpy` installed to run the script.

# Run
run the following command to see the step-by-step visualization.
```
$ python agent.py
```
