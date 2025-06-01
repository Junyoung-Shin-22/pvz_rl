from pvz import config
from agents import evaluate
from agents import ACAgent3, TrainerAC3

import os
import os.path as osp
import glob

_key = lambda x: int(osp.splitext(osp.basename(x))[0])
checkpoints_path = './checkpoints'

if __name__ == "__main__":
    checkpoints = sorted(os.listdir(checkpoints_path))
    
    for checkpoint in checkpoints:
        print(checkpoint)

        policy_net_path = sorted(glob.glob(osp.join(checkpoints_path, checkpoint, 'policy', '*.pt')), key=_key)[-1]
        value_net_path = sorted(glob.glob(osp.join(checkpoints_path, checkpoint, 'value', '*.pt')), key=_key)[-1]

        env = TrainerAC3(max_frames = 500*config.FPS)
        agent = ACAgent3(
                input_size=env.num_observations(),
                possible_actions=env.get_actions(),
                device='cuda'
        )
        agent.load(policy_net_path, value_net_path)
        
        avg_score, avg_iter = evaluate(env, agent, verbose=False)
        
        with open(osp.join(checkpoints_path, checkpoint, 'eval.txt'), 'wt') as f:
            f.write(f"Mean score {avg_score}\n")
            f.write(f"Mean frames {avg_iter}")
