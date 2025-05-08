import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.Building import Building  # Your custom env

# 1) Set up TensorBoard log directory
TB_LOG_DIR = "./tensorboard_logs/"

def make_env():
    return Building(
        total_elevator_num=2,          
        max_floor=6,                  
        max_passengers_in_floor=7,    
        max_passengers_in_elevator=8, 
        elevator_capacity=8,          
        render_mode="human"         
    )

# Vectorize & normalize
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# 2) Pass tensorboard_log to the constructor
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=1000,
    gamma=0.99,
    tensorboard_log=TB_LOG_DIR,     # ← enable TB logging here
)

# 3) When calling learn, give each run a name
model.learn(
    total_timesteps=100_000,
    tb_log_name="ppo_elevator_run"  # ← this subfolder will appear under tensorboard_logs/
)

# 4) Save everything as before
model.save("ppo_elevator")
env.save("vec_normalize.pkl")
