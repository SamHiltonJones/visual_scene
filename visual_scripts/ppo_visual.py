import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
import torch as th
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Callable
from gym import spaces
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class TupleToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TupleToBoxWrapper, self).__init__(env)

        single_space = env.observation_space.spaces[0]
        num_cameras = len(env.observation_space.spaces)

        obs_shape = (*single_space.shape[:2], num_cameras * single_space.shape[2])

        self.observation_space = spaces.Box(
            low=single_space.low.min(),
            high=single_space.high.max(),
            shape=obs_shape,
            dtype=single_space.dtype
        )

    def observation(self, observation):
        stacked_obs = np.concatenate(observation, axis=-1)  
        return stacked_obs

def create_env(env_path, worker_id=0, time_scale=2.0, no_graphics=False):
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(env_path, side_channels=[channel], worker_id=worker_id, no_graphics=no_graphics)
    channel.set_configuration_parameters(time_scale=time_scale)
    
    gym_env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    
    gym_env = TupleToBoxWrapper(gym_env)
    
    return gym_env

def get_save_path(base_path, model_name, trained_path=False):
    date_str = datetime.now().strftime("%d%m%Y")
    version = 0
    if trained_path:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
    else:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
    return save_path[:-4] if trained_path else save_path

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_interval, base_path, model_name, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.base_path = base_path
        self.model_name = model_name
        self.step_count = 0
        self.total_reward = 0

        self.log_file = self.get_log_file_path()

    def get_log_file_path(self):
        save_path = get_save_path(self.base_path, self.model_name)
        os.makedirs(save_path, exist_ok=True)
        log_file = f"{save_path}/{self.model_name}_reward_log.txt"
        version = 0
        while os.path.exists(log_file):
            version += 1
            log_file = f"{save_path}/{self.model_name}_reward_log_v{version}.txt"
        return log_file

    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_reward += np.sum(self.locals['rewards'])

        if self.step_count % self.log_interval == 0:
            with open(self.log_file, 'a') as f:
                f.write(f'Step: {self.step_count}, Reward: {self.total_reward}\n')
            print(f'Logged reward at step {self.step_count}: {self.total_reward}')
            self.total_reward = 0
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[2]  

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(np.zeros((1, n_input_channels, *observation_space.shape[:2]))).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        if observations.shape[3] == 1:
            observations = observations.squeeze(3)
        
        if len(observations.shape) == 4:
            observations = observations.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(f"Unexpected shape for observations: {observations.shape}, expected 4 dimensions.")
        
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),  
    activation_fn=th.nn.ReLU,
    net_arch=[dict(pi=[128, 256, 128], vf=[128, 256, 128])]  
)


if __name__ == '__main__':
    n_envs = 1  

    env = r"C:\methods_compare\builds\post_sub_builds\visual_scene\3DPos.exe"

    def make_env_scene1():
        return create_env(env, worker_id=0, time_scale=4.0, no_graphics=True)

    env_scene1 = DummyVecEnv([make_env_scene1])

    base_path = 'post_sub/visual/logs_models'
    trained_models_path = 'post_sub/visual/trained_models'
    os.makedirs(trained_models_path, exist_ok=True)

    model_name = 'ppo_visual'
    save_path_scene1 = get_save_path(base_path, model_name)
    checkpoint_callback_scene1 = CheckpointCallback(save_freq=50000, save_path=save_path_scene1, name_prefix=model_name)

    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)
    reward_logging_callback_scene1 = RewardLoggingCallback(log_interval=100, base_path=base_path, model_name=model_name)

    model = PPO(
        "CnnPolicy",
        env_scene1,
        verbose=2,
        tensorboard_log=tensorboard_log_path_scene1,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=64,  
        device=th.device("cuda" if th.cuda.is_available() else "cpu")
    )


    model.learn(
        total_timesteps=1000000, reset_num_timesteps=True, tb_log_name="train_scene3",
        callback=[checkpoint_callback_scene1, reward_logging_callback_scene1]
    )

    final_model_path = get_save_path(trained_models_path, model_name, trained_path=True)
    model.save(final_model_path)

    env_scene1.close()
