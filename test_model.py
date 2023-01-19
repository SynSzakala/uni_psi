import sys
from time import sleep

from stable_baselines3 import DQN, PPO

from env.snake_utils import Size
from env.snake_env import SnakeEnv

size = Size(10, 10)
env = SnakeEnv(size)

model_file = "model.zip" if len(sys.argv) < 2 else sys.argv[1]

model = DQN.load(model_file)

obs = env.reset()
while True:
    sleep(0.125)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(chr(27) + "[2J")
    env.render()
    if done:
        obs = env.reset()
