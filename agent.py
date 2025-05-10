from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from environment.Building import Building
from stable_baselines3.common.logger import configure
import environment.utils as utils

def make_env():
    return Building(
        elevator_count=2,
        max_floor=6,
        floor_capacity=8,
        elevator_capacity=8,
        render_mode="human",
        #step_gen_flag=True
    )


class CumulativeRewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 'infos' is a list of dicts, one per env
        for info in self.locals["infos"]:
            # When a new episode ends, VecMonitor injects an 'episode' dict
            if "episode" in info:
                # Log the total reward from this episode
                ep_reward = info["episode"]["r"]
                self.logger.record("episode/cumulative_reward", ep_reward)
        return True


# (Re)build the env as shown above: DummyVecEnv → VecMonitor → VecNormalize
env = DummyVecEnv([make_env])
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Instantiate PPO with TensorBoard logging enabled
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=1e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=utils.TB_LOG_DIR,
    # use_sde=False,            # State-dependent exploration off by default
    # sde_sample_freq=-1,       # Only if you enable SDE
)

new_logger = configure(
    utils.TB_LOG_DIR, 
    ["stdout", "csv", "tensorboard"]
)
model.set_logger(new_logger)

# Train with our callback
for epoch in range(1, utils.EPOCHS+1):
    model.learn(
        total_timesteps = utils.HORIZONS,
        reset_num_timesteps = False,
        tb_log_name = f"ppo_elevator2_run",
        callback = CumulativeRewardLogger()
    )
    if(epoch % 50 == 0):
        print("Done with model training for epoch: ", epoch)

# Save model & normalization stats
model.save("ppo_elevator2")
env.save("vec_normalize2.pkl")