import gym
import time
from stable_baselines3 import A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
from typing import Callable


# create new environment for multi-processing
def make_env(env_id: str, rank: int, seed:  int = 0) -> Callable:
    def _init() -> gym.Env:
        env = gym.make(env_id)  # create the environment
        env.seed(seed+rank)  # give a random seed to the environment
        return env
    set_random_seed(seed)  # assign the new random seed
    return _init


def main():
    env_id = "LunarLander-v2"  # The environment to use
    num_cpu = 8  # number of cpu cores to use
    logdir = "logs"  # the name of the folder the information is saved in
    modeldir = "bestModelA2CTuned"  # folder where the best model is saved

    # create the save folder if it doesn't exist
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # create save folder for the model as well
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # create the environments using multiprocessing
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # create the agent
    model = A2C("MlpPolicy", vec_env, verbose=0,
                tensorboard_log=logdir, ent_coef=1e-2,
                learning_rate=5e-4)

    eval_callback = EvalCallback(vec_env,
                                 best_model_save_path=modeldir,
                                 log_path=logdir, eval_freq=10000,
                                 deterministic=True,
                                 render=False)
    # train the agent
    _timesteps = 32_000
    for i in range(1, 30):

        # time wrapper to understand how long a learning cycle has taken
        start_time = time.time()
        # train the agent
        model.learn(total_timesteps=_timesteps,
                    reset_num_timesteps=False, tb_log_name="A2C_tuned2",
                    callback=eval_callback)
        # print time
        total_time_multi = time.time() - start_time
        print(f"Took {total_time_multi:.2f}s for multiprocessed version - {_timesteps / total_time_multi:.2f} FPS")


if __name__ == "__main__":
    main()
