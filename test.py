# test.py
import time
from stable_baselines3 import PPO
from Building import Building

def run_trained_model(
    model_path='models/final_ppo_elevator.zip',
    total_elevators=2,
    max_floor=10,
    max_passengers_in_floor=5,
    max_passengers_in_elevator=4,
    elevator_capacity=10,
    spawn_prob=0.1,
    max_steps=200
):
    # Load the trained model
    model = PPO.load(model_path)

    # Create environment for evaluation
    env = Building(
        total_elevator_num=total_elevators,
        max_floor=max_floor,
        max_passengers_in_floor=max_passengers_in_floor,
        max_passengers_in_elevator=max_passengers_in_elevator,
        elevator_capacity=elevator_capacity
    )

    obs = env.reset()
    # Ensure at least one passenger to start episode
    waiting = obs['waiting'].sum()
    if waiting == 0:
        env.generate_passengers(env.spawn_prob)
        obs = env._build_obs()
    for step in range(max_steps):
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        # Step environment
        obs, reward, done, info = env.step(action)

        # Render building state
        print(f"Step {info['step']} | Reward: {reward}")
        env.render()

        time.sleep(0.5)  # pause for visualization
        if done:
            print("All passengers served, ending episode.")
            break

    env.close()

if __name__ == '__main__':
    run_trained_model()
