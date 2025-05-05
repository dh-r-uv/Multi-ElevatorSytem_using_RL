import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from environment.Building import Building

def make_env():
    # Create the Building environment with the same parameters used during training
    return Building(
        total_elevator_num=2,       # Adjust based on your training setup
        max_floor=10,              # Adjust based on your training setup
        max_passengers_in_floor=20, # Adjust based on your training setup
        max_passengers_in_elevator=10, # Adjust based on your training setup
        elevator_capacity=10,       # Adjust based on your training setup
        render_mode="human"       # Set to "human" for rendering
    )

# Set up the environment
env = make_env()
env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for compatibility
env = VecNormalize.load("vec_normalize.pkl", env)  # Load normalization stats

# Load the trained PPO model
model = PPO.load("ppo_elevator", env=env)

# Test for a specified number of episodes
num_episodes = 10

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
        env.render()
        time.sleep(0.1)  # Add a delay to make rendering observable
        step += 1
    # Print episode statistics
    # print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {step}, Arrived Passengers = {info[0]['arrived_passengers']}")

# Note: Ensure "ppo_elevator.zip" and "vec_normalize.pkl" are in the current directory or adjust paths accordingly