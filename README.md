Finally got around to uploading this!
This is an AI agent that can solve stochastic Sokoban puzzles using a heuristic.

What’s the heuristic, you may wonder?
First, we use an AI agent (also created by yours truly) to solve a deterministic version of the puzzle.
We then try to replicate the same moves in the stochastic version. If the agent deviates, and the deviation is only by a few steps, it tries to recover by returning to the correct position.

If it can’t recover within a few steps, it restarts the whole process — beginning again from solving the deterministic version.

To run it, just execute one of the check files.
It goes without saying that, since the environment is stochastic, the agent might not solve all puzzles within the time limit. You can tweak the parameters as you wish and check the results.
