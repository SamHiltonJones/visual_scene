import matplotlib.pyplot as plt
import pandas as pd

def read_data(filename):
    steps = []
    rewards = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            step = int(parts[0].split(':')[1].strip())
            reward = float(parts[1].split(':')[1].strip())
            steps.append(step)
            rewards.append(reward)
    return steps, rewards

def moving_average(data, window_size):
    return pd.Series(data).rolling(window=window_size).mean()

def plot_moving_average(steps, rewards, window_size, label):
    moving_avg_rewards = moving_average(rewards, window_size)
    plt.plot(steps, moving_avg_rewards, label=label)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward vs. Step with Moving Average')

def main():
    filenames = [
        r'post_sub\visual\logs_models\ppo_visual_08102024_v0\ppo_visual_reward_log.txt'
    ]

    window_size = 100

    for filename in filenames:
        steps, rewards = read_data(filename)
        label = filename.split("\\")[-2] 
        plot_moving_average(steps, rewards, window_size, label)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
