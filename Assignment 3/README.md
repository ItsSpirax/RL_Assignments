## RTDP (decaying epsilon) + MCTS — Assignment

Your job: fill the sections marked `#YOUR CODE HERE` in `rtdp.py` and `mcts.py`.

### Tasks (do these):
1. RTDP: implement `bellman_backup` and the episode loop with decaying epsilon-greedy.
2. MCTS (UCT): implement one iteration (selection, expansion, rollout, backprop) and return the most visited action.
3. Run both on the default GridWorld and print steps-to-goal and total reward for at least 20 episodes/searches.
4. Briefly (3–5 sentences) compare RTDP vs MCTS behavior on this map.
5. Optional: try different epsilon schedules and `c_uct` values and comment.

### Run
- Edit `main.py` and uncomment `run_rtdp()` or `run_mcts()`.
- `python main.py`
![alt text](image.png)

### Hints
- RTDP: `V[s] = max_a E[r + gamma V[s']]` using the provided model. Use epsilon-greedy over one-step lookahead Q(s,a).
- MCTS: UCT score `Q + c * sqrt(ln N / (1 + N_a))`. Discount returns in rollout/backprop.

