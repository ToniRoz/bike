# Bike Wheel Truing Reinforcment learning framework

Contains a RL-Environment that simulates the truing process of a spoked bycicle wheel ,as well as, different RL-algorithms to find a scalable automatic solution through training.

## General

The simulation was taken and only slightly runtime-optimized from the dissertation on bycicle-wheels from Dr. Matthew Ford.
A web-implementation, python code and information on the dissertation can be found here: [1]
There are three different Algorithms implemented in this repo: DQN (mostly with all the Rainbow improvements)[2], PPO[3] and TD-MPC2[4]
All of them have a hydra-integrated config, where Rainbow and PPO link to the Environment config in the root/configs folder, where as TD-MPC2 instantiates from a seperate config in the tdmpc-components dir.

## Algorithm Implementation

### DQN (Rainbow)

Was originally taken from this implementation[5]
I added:
    - Implicit Quantile support as described in [8]
    - Support for recurrent Networks namely lstm and gru 

### PPO

Was originally taken from this implementation[6]
I added:
    - Support for recurrent Networks namely lstm and gru 

### TDMPC2

Was originally taken from this implementation[7]
which was mostly taken as is, just adding some network parameters to the config

## Usage

The scripts, </br></br>

/tdmpc-components/train.py (TD-MPC2)</br></br>
train_ppo.py (PPO)</br></br>
train_rainbow.py (RAINBOW)</br></br>

serve as quick entriepoints to start a training on the respective algorithms. ppo_- and rainbow_exp.py show how to run longer experiments (hyper-parameter gridsearch).
For a more Hydra-instanciation usage check out the colab-notebook.


## References
[1] [Matthew Ford Github](https://github.com/dashdotrobot)
[2][Rainbow Paper](https://arxiv.org/abs/1710.02298)
[3][PPO Paper](https://arxiv.org/abs/1707.06347)
[4][TDMPC2](https://www.tdmpc2.com/)
[5][Rainbow Git](https://github.com/Kaixhin/Rainbow)
[6][PPO implementation](https://github.com/saqib1707/RL-PPO-PyTorch)
[7][TDMPC2 implementation](https://github.com/ShaneFlandermeyer/tdmpc2-jax)
[8][Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
