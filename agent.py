import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.Building import Building

def train_ppo(
    total_elevators=2,
    max_floor=10,
    max_passengers_in_floor=5,
    max_passengers_in_elevator=4,
    elevator_capacity=10,
    spawn_prob=0.1,
    n_envs=4,
    timesteps=200_000,
    save_dir='./models'
):
    os.makedirs(save_dir, exist_ok=True)
    env = make_vec_env(
        lambda: Building(
            total_elevators,
            max_floor,
            max_passengers_in_floor,
            max_passengers_in_elevator,
            elevator_capacity
        ),
        n_envs=n_envs
    )
    callback = CheckpointCallback(
        save_freq=timesteps // 10,
        save_path=save_dir,
        name_prefix='ppo_elevator'
    )
    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    final_path = os.path.join(save_dir, 'final_ppo_elevator')
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}.zip")
    return model

if __name__ == '__main__':
    train_ppo()