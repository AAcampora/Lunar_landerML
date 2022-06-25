
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

env_id = "LunarLander-v2"  # The environment to use

env = gym.make(env_id)
model = PPO.load("bestModel2/best_model.zip")
env = Monitor(env, "logs")
evaluate_policy(model, env, n_eval_episodes=20, render=True)

# episodes = 10
# for ep in range(episodes):
#     obs = env.reset()
#     dones = False
#     while not dones:
#         env.render()
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)


env.close()