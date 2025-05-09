import time
import gym
import environment.utils as utils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from environment.Building import Building

def make_env():
    return Building(
        elevator_count=2,
        max_floor=6,
        floor_capacity=7,
        # max_passengers_in_elevator=8,
        elevator_capacity=8,
        render_mode="ansi"
    )

# def make_env():
#     return Building(
#         elevator_count=2,
#         max_floor=4,
#         floor_capacity=5,
#         # max_passengers_in_elevator=8,
#         elevator_capacity=5,
#         render_mode="human"
#     )

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--flag",
    action="store_true",   # sets flag=True if --flag is present
    help="turn on the flag (default: off)"
)
args = parser.parse_args()

flag = args.flag   # False if omitted, True if --flag is passed

passengers = []
if(flag):
    with open("input.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            x, y = map(int, line.split())
            x-=1
            y-=1
            passengers.append([x, y])

print(flag, passengers)

# Set up the environment
env = make_env()
env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for compatibility
env = VecNormalize.load("vec_normalize_final.pkl", env)  # Load normalization stats

# Load the trained PPO model
model = PPO.load("ppo_elevator_final", env=env)

# Test for a specified number of episodes
num_episodes = 1

for episode in range(num_episodes):
    env.envs[0].set_flag(flag, passengers)  # Set the flag and passengers for the environment
    obs = env.reset()  # Reset the environment
    done = [False]
    episode_reward = 0
    step = 0
    env.envs[0].render()  # Initial render
    while not done[0]:
        # Predict the next action using the trained model
        action, _ = model.predict(obs)
        # Step the environment with the predicted action
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        # Render the environment to visualize the state
        env.envs[0].render()
        time.sleep(1)  # Add a delay to make rendering observable
        step += 1
    # Print episode statistics
    # print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {step}, Arrived Passengers = {info[0]['arrived_passengers']}, Remaining Passengers = {info[0]['remaining_passengers']}")

# Note: Ensure "ppo_elevator.zip" and "vec_normalize.pkl" are in the current directory or adjust paths accordingly