import os
import numpy as np

import tqdm
import datetime

def timestamp():
    return datetime.datetime.now().strftime('%y%m%d_%H%M%S')

def train(env, agent, n_iter=100000, n_record=10000):
    ts = timestamp()
    checkpoint_path = f'checkpoints/{ts}/'
    
    os.makedirs(os.path.join(checkpoint_path, 'policy'))
    os.makedirs(os.path.join(checkpoint_path, 'value'))

    sum_score = 0
    best_score = 0

    sum_iter = 0
    score_list = []
    iter_list = []

    pbar = tqdm.trange(0, n_iter, desc='mean_score: N/A')
    for i in pbar:
        # play episodes
        summary = env.play(agent)
        rewards = summary["rewards"]
        score = np.sum(rewards)

        sum_score += score
        sum_iter += min(env.env._scene._chrono, env.max_frames)

        # Update agent
        agent.update(rewards)

        if ((i + 1) % n_record == 0):
            if sum_score >= best_score:
                agent.save(f'{checkpoint_path}/policy/{i}.pt',
                           f'{checkpoint_path}/value/{i}.pt')
                best_score = sum_score

            pbar.set_description(f'mean_score: {sum_score/n_record}')

            score_list.append(sum_score/n_record)
            iter_list.append(sum_iter/n_record)
            sum_iter = 0
            sum_score = 0

    np.savez(f'{checkpoint_path}/train.npz', score=score_list, iter=iter_list)


from agents import ACAgent3, TrainerAC3

if __name__ == "__main__":

    env = TrainerAC3(max_frames = 1000)
    agent = ACAgent3(
        input_size = env.num_observations(),
        possible_actions=env.get_actions(),
        gamma = 0.85,
        device='cuda'
    )
    train(env, agent)




