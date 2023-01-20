import sys
from time import sleep

from stable_baselines3 import DQN, PPO

from env.snake_utils import Size
from env.snake_env import SnakeEnv

size = Size(10, 10)
env = SnakeEnv(size)

model_file = "model.zip" if len(sys.argv) < 2 else sys.argv[1]

model = DQN.load(model_file)

len_sum = 0
rew_sum = 0
total_episodes = 0

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if reward > 0:
        rew_sum += reward
    len_sum += 1
    if done:
        total_episodes += 1
        if total_episodes == 1000:
            print(
                f"1000 episodes, avg len: {len_sum / total_episodes}, avg rew/episode {rew_sum / total_episodes},"
                f" total rew/len {rew_sum / len_sum}"
            )
            exit(0)
        obs = env.reset()

