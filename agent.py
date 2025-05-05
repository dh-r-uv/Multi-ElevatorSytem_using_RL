import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.Building import Building  # Assuming Building class is in environment/Building.py

# Step 1: Create the Building environment
def make_env():
    return Building(
        total_elevator_num=2,          # Number of elevators
        max_floor=10,                  # Number of floors
        max_passengers_in_floor=20,    # Max passengers per floor
        max_passengers_in_elevator=10, # Max passengers per elevator
        elevator_capacity=10,          # Elevator capacity
        render_mode="human"         # Render mode for visualization
    )

# Vectorize the environment for compatibility with Stable Baselines3
env = DummyVecEnv([make_env])

# Normalize observations to improve training stability
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Step 2 & 3: PPO setup handles emptying the building and generating passengers via reset
# Step 4: PPO's learn method performs actions, checks rewards, and updates the policy
model = PPO(
    "MultiInputPolicy",  # Policy suitable for dictionary observation spaces
    env,
    verbose=1,
    learning_rate=3e-4,  # Learning rate for optimization
    n_steps=2048,        # Number of steps per update
    batch_size=64,       # Minibatch size for policy updates
    n_epochs=10,         # Number of epochs per update
    gamma=0.99           # Discount factor for rewards
)

# Train the model
total_timesteps = 100000  # Total training steps; adjust as needed
model.learn(total_timesteps=total_timesteps)

# Save the trained model and normalization parameters
model.save("ppo_elevator")
env.save("vec_normalize.pkl")

# Optional: Load and test the model
env = DummyVecEnv([make_env])
env = VecNormalize.load("vec_normalize.pkl", env)
model = PPO.load("ppo_elevator", env=env)