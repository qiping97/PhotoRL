# PhotoRL

## Data Collection using Shutter

3 log files output each timestep.

`discretized_mdp_start_<timestamp>.json`: will be used to derive state, action, and reward for RL. a list of step in one experiment, each element in the list is a list containing the original information about the state, action and reward. state is a dictionary composed of `num_photo`, `position`, 'start_new_photo', `user_position` information. action is an integer representing an action chosen from 101 possible actions. reward is a list consist 9 values, each answers for one survey question.

`mdp_start_<timestamp>.json`:

`raw_start_<timestamp>.json`:


## Reinforcement Learning 


### Episode

Based on `discretized_mdp_start_<timestamp>.json`, we extract state, action and reward as follows,

state: a tuple with 6 elements:

 1. subpolicy: 4 types - rear depth and low position, rear depth and high position, middle depth, forward depth
 2. position: position of shutter, 93 possible integers, each integer index mapps to an array with 4 elements representing the angle of each joint
 3. start_new_photo: boolean, True: new photo, False: old one
 4. UserPosition[0]
 5. UserPosition[1]
 6. UserPosition[2]

action: an integer is used to represent one action selected from 101 candidate actions including transition and "taking photo"

reward: a list of 9 elements representing the evaluation from users. 0 meaning the question is not answered, values larger than 0 will be scaled to a new range [-4,4].


### Q-value Table

If haven't met the state and action, default Q value (i.e. 0) will be filled in.


### Off-Polocy Monte Carlo Method

(parameters set up:)

(...)


