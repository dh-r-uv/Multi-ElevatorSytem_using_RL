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
        floor_capacity=8,
        elevator_capacity=8,
        render_mode="human"
    )


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--flag",
    action="store_true",   # sets flag=True if --flag is present
    help="turn on the flag (default: off)"
)

parser.add_argument(
    "--step_gen_flag",
    action="store_true",   # sets flag=True if --flag is present
    help="turn on the flag (default: off)"
)
args = parser.parse_args()

flag = args.flag   # False if omitted, True if --flag is passed
step_gen_flag = args.step_gen_flag   # False if omitted, True if --flag is passed

passengers = []
if(flag):
    with open("input.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            x, y = map(int, line.split())
            x-=1
            y-=1
            passengers.append([x, y])


# Set up the environment
env = make_env()
env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for compatibility
env = VecNormalize.load("2 Elevators/vec_normalize.pkl", env)  # Load normalization stats

# Load the trained PPO model
model = PPO.load("2 Elevators/ppo_elevator.zip", env=env)

# Test for a specified number of episodes
num_episodes = 2


for episode in range(num_episodes):
    env.envs[0].set_flag(flag, passengers, step_gen_flag)  # Set the flag and passengers for the environment
    obs = env.reset()  # Reset the environment
    done = [False]
    episode_reward = 0
    step = 0
    print("\033[H\033[J", end="")
    env.envs[0].render()  # Initial render
    time.sleep(1)
    while not done[0]:
        # Predict the next action using the trained model
        action, _ = model.predict(obs)
        # Step the environment with the predicted action
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        if(done[0]):
            time.sleep(5)
            continue
        # Render the environment to visualize the state
        print("\033[H\033[J", end="")
        env.envs[0].render()
        

        time.sleep(1)  # Add a delay to make rendering observable
        step += 1

# Note: Ensure "ppo_elevator.zip" and "vec_normalize.pkl" are in the current directory or adjust paths accordingly