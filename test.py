import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from environment.Building import Building

def make_env():
    return Building(
        total_elevator_num=2,          # Number of elevators
        max_floor=6,                  # Number of floors
        max_passengers_in_floor=7,    # Max passengers per floor
        max_passengers_in_elevator=8, # Max passengers per elevator
        elevator_capacity=8,          # Elevator capacity
        render_mode="human"         # Render mode for visualization
    )

# Set up the environment
env = make_env()
env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for compatibility
env = VecNormalize.load("vec_normalize.pkl", env)  # Load normalization stats

# Load the trained PPO model
model = PPO.load("ppo_elevator", env=env)

# Test for a specified number of episodes
num_episodes = 5

for episode in range(num_episodes):
    obs = env.reset()
    done = [False]
    episode_reward = 0
    step = 0
    while not done[0]:
        # Predict the next action using the trained model
        action, _ = model.predict(obs)
        # Step the environment with the predicted action
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        # Render the environment to visualize the state
        env.envs[0].render()
        time.sleep(0.5)  # Add a delay to make rendering observable
        step += 1
    # Print episode statistics
    print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {step}, Arrived Passengers = {info[0]['arrived_passengers']}, Remaining Passengers = {info[0]['remaining_passengers']}")

# Note: Ensure "ppo_elevator.zip" and "vec_normalize.pkl" are in the current directory or adjust paths accordingly