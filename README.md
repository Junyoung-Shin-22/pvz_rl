# pvz_rl
Plants vs. Zombies remake and RL agents

## Installation/Setup
The code was executed using an Anaconda environment with Python 3.9.21.
To run this code, use the Anaconda command prompt (terminal) associated with that environment.
Use the command prompt to move into the directory where the code file is located.

By running the following commands:
cd gym-pvz
pip install -e .

the package is installed in editable (development) mode,
which allows you to modify the original source code and have the changes reflected immediately without reinstalling.


## Requirements
The following python libraries are required: pytorch, gym, pygame (and shap to evaluate feature importance if wanted).
To install all required dependencies at once, you can use the provided requirements.txt file:
pip install -r requirements.txt
This will ensure that all necessary packages are installed with the appropriate versions.



## Usage example
Here we will guide you to train an agent from scratch. However, since the training can be fairly long depending on your machine (more than 30 minutes if the agent performs well), you can skip the training section (you then do not have to modify any of the code which will then use a pretrained agent from our agent zoo).

### Train an agent
To train a DDQN agent

```
python train_ddqn_agent.py
```
To train other agents, use the dedicated train scripts. You will be asked to save the agent under a certain name/path we will refer as NAME in 
During the training process, the loss, reward, and frame information for each episode is saved as files in the directory corresponding to the previously entered name.
If desired, this data can be visualized by writing a plotting script in either MATLAB or Python.



### Visualize a play
With the following, you can visualize a game of an agent.
```
python game_render.py
```
By default, this will show the behavior of the DDQN agent. You can modify agent_type in game_render.py to use some other saved agents or even your own agent (doing the exact same modifications as above)

To visualize a game with higher FPS (more fluid motions), change the FPS variable in pvz/pvz/config.py. This may have a slight impact on the behavior of some agents.
