import matplotlib
import numpy as np
import sys
import json
from os import listdir
from os.path import isfile, join
from collections import defaultdict


def load_episodes_from_logs(data_path):
    discretized_mdp_files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) and f.startswith('discretized')]
    episodes = []

    for discretized_mdp_file in discretized_mdp_files:
        f = open(discretized_mdp_file)
        data = json.load(f)
        episode = []
        photo_idx = -1

        for i in range(len(data)):
            timestep = data[i]
            state = timestep[0]
            action = timestep[1]
            reward = timestep[2]

            photo_idx = state["num_photos"]

            s = tuple([state["subpolicy"], state["position"], int(state["starting_new_photo"] == True), state["user_position"][0], state["user_position"][1], state["user_position"][2]])
            a = action
            r = []

            for raw_reward in reward:
                scaled_reward = raw_reward - 4 if raw_reward != 0 else 0
                r.append(scaled_reward)

            r = tuple(r)

            episode.append((s, a, r))

            if i == len(data)-1 or data[i+1][0]["starting_new_photo"]:
                episodes.append(episode)
                episode = []

    return episodes


def compute_behavior_policy(subpolicy_to_num_positions, state, action):
    if action == 100:
        return 1./3
    else:
        return 2./3*1./subpolicy_to_num_positions[state[0]]


def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn


def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn


def mc_control_importance_sampling(episodes, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """

    action_space_size = 101
    num_rewards = 9
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    

    Qs = []
    for i in range(num_rewards):
        Q = defaultdict(lambda: np.zeros(action_space_size))
        Qs.append(Q)

    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    Cs = []
    for i in range(num_rewards):
        C = defaultdict(lambda: np.zeros(action_space_size))
        Cs.append(C)
    
    # Our greedily policy we want to learn

    for i_reward in range(num_rewards):
        print("Computing Q table for reward " + str(i_reward))
        target_policy = create_greedy_policy(Qs[i_reward])
            
        for i_episode in range(1, len(episodes) + 1):
            # Print out which episode we're on, useful for debugging.
            print("Episode {}/{}.".format(i_episode, len(episodes)))

            episode = episodes[i_episode-1]
            
            # Sum of discounted returns
            G = 0.0
            # The importance sampling ratio (the weights of the returns)
            W = 1.0
            # For each step in the episode, backwards
            for t in range(len(episode))[::-1]:
                state, action, reward = episode[t]
                # Update the total reward since step t
                G = discount_factor * G + reward[i_reward]
                # Update weighted importance sampling formula denominator
                Cs[i_reward][state][action] += W
                # Update the action-value function using the incremental update formula (5.7)
                # This also improves our target policy which holds a reference to Q
                Qs[i_reward][state][action] += (W / Cs[i_reward][state][action]) * (G - Qs[i_reward][state][action])

                target_policy = create_greedy_policy(Qs[i_reward])

                # If the action taken by the behavior policy is not the action 
                # taken by the target policy the probability will be 0 and we can break
                if action !=  np.argmax(target_policy(state)):
                    break
                W = W * 1./compute_behavior_policy(subpolicy_to_num_positions, state, action)
        
    return Qs

subpolicy_to_num_positions = {0: 21, 1: 21, 2: 24, 3: 27}

#data_path = '/home/qz257/Projects/PhotoRL/shutter_tarzan/shutter-tarzan-shutter_rl/shutter_tarzan/rl_data_temp_updated/'
#output_path = 'Q_table.json'

data_path = '/home/aon7/tarzan_ws/src/shutter-tarzan/shutter_tarzan/rl_data_all/'
output_path = 'Q_table_all_data2.json'

episodes = load_episodes_from_logs(data_path)
Qs = mc_control_importance_sampling(episodes)

Qs_json = []

for i in range(len(Qs)):
    Q_json = {}
    for key, value in Qs[i].items():
        Q_json[str(key)] = list(value)
    Qs_json.append(Q_json)

with open(output_path, 'w') as outfile:
    json.dump(Qs_json, outfile)