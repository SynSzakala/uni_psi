from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn.policies import MultiInputPolicy, DQNPolicy
from stable_baselines3.ppo import MlpPolicy, CnnPolicy

from custom_combined_extractor import CustomCombinedExtractor, CustomNatureCNN
from env.snake_utils import Size
from env.snake_env import SnakeEnv
from snake_feature_extractor import SnakeFeatureExtractor

size = Size(10, 10)
env = SnakeEnv(size)

model: PPO

try:
    model = PPO.load(
        "model.zip",
        env=env,
        verbose=1,
        tensorboard_log='./tb'
    )
except:
    model = PPO(
        policy=CnnPolicy,
        policy_kwargs={
            "net_arch": [256, 256],
            "features_extractor_class": CustomNatureCNN,
        },
        env=env,
        verbose=1,
        tensorboard_log='./tb',
    )

while True:
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name="learn")
    model.save("model.zip")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
