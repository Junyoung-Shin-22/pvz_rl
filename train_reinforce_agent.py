import gym
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pvz import config
import matplotlib.pyplot as plt
import time
import os

# Import your agent

from agents import HJ_ReinforceAgent, PlayerV2


import time
import matplotlib.pyplot as plt

def train(env, agent, n_iter=100000, n_record=500):
    start_time = time.time()

    sum_score = 0
    sum_iter = 0
    score_plt = []
    iter_plt = []

    for episode_idx in range(n_iter):
        summary = env.play(agent)
        score = np.sum(summary["rewards"])
        frame = min(env.env._scene._chrono, env.max_frames)

        sum_score += score
        sum_iter += frame
        agent.update(summary["observations"], summary["actions"], summary["rewards"])

        if (episode_idx % n_record == n_record - 1):
            mean_score = sum_score / n_record
            mean_frame = sum_iter / n_record
            print(f"--- Episode {episode_idx + 1}: mean score = {mean_score:.2f}, mean frames = {mean_frame:.2f}")
            score_plt.append(mean_score)
            iter_plt.append(mean_frame)
            sum_score = 0
            sum_iter = 0

    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time:.2f} seconds")

    os.makedirs("figures", exist_ok=True)

    plt.figure()
    plt.plot(range(n_record, n_iter + 1, n_record), score_plt)
    plt.title("Mean Score over Training")
    plt.xlabel("Episode")
    plt.ylabel("Mean Score")
    plt.grid()
    plt.savefig("figures/train_score_plot.png")
    plt.close()

    plt.figure()
    plt.plot(range(n_record, n_iter + 1, n_record), iter_plt)
    plt.title("Mean Frame over Training")
    plt.xlabel("Episode")
    plt.ylabel("Mean Frame")
    plt.grid()
    plt.savefig("figures/train_frame_plot.png")
    plt.close()

    evaluate(env, agent)

def evaluate(env, agent, n_iter=1000, verbose=True):
    sum_score = 0
    sum_iter = 0
    score_hist = []
    iter_hist = []

    for episode_idx in range(n_iter):
        if verbose:
            print(f"\rEvaluating... {episode_idx + 1}/{n_iter}", end="")
        summary = env.play(agent)
        score = np.sum(summary["rewards"])
        frame = min(env.env._scene._chrono, config.MAX_FRAMES)

        score_hist.append(score)
        iter_hist.append(frame)
        sum_score += score
        sum_iter += frame

    print("\n\n Evaluation Summary:")
    print(f"Mean score over {n_iter} episodes: {sum_score / n_iter:.2f}")
    print(f"Max score over {n_iter} episodes: {np.max(score_hist):.2f}")
    print(f"Mean frame over {n_iter} episodes: {sum_iter / n_iter:.2f}")
    print(f"Max frame over {n_iter} episodes: {np.max(iter_hist)}")

    if verbose:
        os.makedirs("figures", exist_ok=True)
        plt.figure()
        plt.hist(score_hist, bins=30)
        plt.title(f"Score Distribution over {n_iter} episodes")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid()
        plt.savefig("figures/eval_score_hist.png")
        plt.close()

    return sum_score / n_iter, sum_iter / n_iter

if __name__ == "__main__":

    env = PlayerV2(render=False, max_frames = 400)
    agent = HJ_ReinforceAgent(
        input_size=env.num_observations(),
        possible_actions=env.get_actions(),
        use_baseline=False,
        use_entropy=True,
        lambda_entropy=0.1
    )
    # agent.policy = torch.load("saved/policy13_v2")
    train(env, agent, n_iter=100000)