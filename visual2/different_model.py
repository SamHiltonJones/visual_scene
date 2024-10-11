from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from gym import spaces
from Agent import Agent
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import gym
import time
import numpy as np
import os
from datetime import datetime
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityToGymWrapper


class TupleToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TupleToBoxWrapper, self).__init__(env)


        # Assume all observations are of the same shape (84, 84, 1)
        single_space = env.observation_space.spaces[0]
        num_cameras = len(env.observation_space.spaces)


        # New observation space: (84, 84, 10)
        obs_shape = (*single_space.shape[:2], num_cameras * single_space.shape[2])


        self.observation_space = spaces.Box(
            low=single_space.low.min(),
            high=single_space.high.max(),
            shape=obs_shape,
            dtype=single_space.dtype
        )


    def observation(self, observation):
        # Concatenate the observations along the channel axis
        stacked_obs = np.concatenate(observation, axis=-1)  # Shape: (84, 84, 10)
        return stacked_obs




def create_env(env_path, worker_id=1, time_scale=2.0, no_graphics=True):
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(env_path, side_channels=[channel], worker_id=worker_id, no_graphics=no_graphics, base_port=384)
    channel.set_configuration_parameters(time_scale=time_scale)
   
    # Wrap the Unity environment for Gym compatibility
    gym_env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
   
    # Wrap the Gym environment to handle Tuple of Box observations
    gym_env = TupleToBoxWrapper(gym_env)
   
    return gym_env
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


if __name__ == '__main__':
    num_envs = 1  
    agent_name = 'trial'
    game = 'game'

    env = r""

    def make_env_scene1():
        return create_env(env, worker_id=0, time_scale=4.0, no_graphics=True)


    env = DummyVecEnv([make_env_scene1])


    base_path = 'post_sub/visual/logs_models'
    trained_models_path = 'post_sub/visual/trained_models'
    os.makedirs(trained_models_path, exist_ok=True)


    model_name = 'ppo_visual'
    save_path_scene1 = get_save_path(base_path, model_name)
    checkpoint_callback_scene1 = CheckpointCallback(save_freq=50000, save_path=save_path_scene1, name_prefix=model_name)




    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)
    reward_logging_callback_scene1 = RewardLoggingCallback(log_interval=100, base_path=base_path, model_name=model_name)


    base_path = 'picture_models/logs_models'
    trained_models_path = 'picture_models/trained_models'
    os.makedirs(trained_models_path, exist_ok=True)


    model_name = 'BTR'
    save_path_scene1 = get_save_path(base_path, model_name)
    checkpoint_callback_scene1 = CheckpointCallback(save_freq=250000, save_path=save_path_scene1, name_prefix=model_name)


    policy_kwargs = dict(
        features_extractor_class=None,  
        activation_fn=th.nn.ReLU,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )


    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)
    reward_logging_callback_scene1 = RewardLoggingCallback(log_interval=1000, base_path=base_path, model_name=model_name)


    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    agent = Agent(n_actions=6, input_dims=[5, 84, 84], device=th.device('cpu'), num_envs=1,
                agent_name="Visual_Agent", total_frames=200000000, testing=0, batch_size=32, rr=1, lr=1e-4,
                maxpool_size=6, ema=0, trust_regions=0, target_replace=32000, ema_tau=0.01,
                noisy=1, spectral=1, munch=1, iqn=1, double=0, dueling=1, impala=1,
                discount=0.997, adamw=0, discount_anneal=0,
                per=1, taus=8, model_size=0.5, linear_size=64,
                ncos=64, maxpool=0, replay_period=4,
                analytics=0, pruning=0, framestack=1, arch="impala",
                per_alpha=0.2, per_beta_anneal=0, layer_norm=1,
                c51=0)
   
    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for i in range(num_envs)]
    scores = []
    observation = env.reset()


    if isinstance(observation, np.ndarray):
        observation = torch.tensor(observation)
       
        if len(observation.shape) == 4:
            observation = observation.permute(0, 3, 1, 2)  


    processes = []
    while steps < 200000000:
        steps += num_envs
        action = agent.choose_action(observation)
        # env.step_async(action)
        observation_, reward, done_, info = env.step(action)


        if isinstance(observation_, np.ndarray):
            observation_ = torch.tensor(observation_)
       
        if len(observation_.shape) == 4:  
            observation_ = observation_.permute(0, 3, 1, 2)  


        agent.learn()
        # observation_, reward, done_, trun_, info = env.step_wait()
        # done_ = np.logical_or(done_, trun_)


        for i in range(num_envs):
            scores_count[i] += reward[i]
            if done_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0


        # reward = np.clip(reward, -1., 1.)


        for stream in range(num_envs):
            terminal_in_buffer = done_[stream]
            agent.store_transition(observation[stream], action[stream], reward[stream], observation_[stream],
                                    terminal_in_buffer, stream=stream)


        observation = observation_


        if steps % 100 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                        .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time)),
                        flush=True)
                last_steps = steps
                last_time = time.time()


    env.close()



