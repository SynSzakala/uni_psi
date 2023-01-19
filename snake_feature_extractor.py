import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict

from env.snake_utils import Size
from env.snake_env import SnakeEnv


class SnakeFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, size: Size):
        super().__init__(observation_space, features_dim=64)
        self.size = size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        #with th.no_grad():
        #    tiles = th.as_tensor(observation_space.sample()["tiles"])
        #    n_flatten = self.cnn(self.__to_2d(tiles).float()).shape[0]
#
        #self.linear = nn.Sequential(nn.Linear(n_flatten, size.area), nn.ReLU())

    def forward(self, observations: TensorDict) -> th.Tensor:
        tiles_2d = self.__to_2d(observations["tiles"])
        extracted_tiles: th.Tensor = self.linear(self.cnn(tiles_2d))
        return th.cat([extracted_tiles, observations["direction"]])

    def __to_2d(self, tiles: th.Tensor) -> th.Tensor:
        return th.reshape(tiles, (1, self.size.height, self.size.width))

