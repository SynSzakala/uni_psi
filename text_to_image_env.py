import numpy as np
from PIL import Image, ImageDraw
from gym.core import Env, Wrapper


class TextToImageEnv(Wrapper):
    metadata = {"render.modes": ['rgb_array']}

    def __init__(self, env: Env, width: int, height: int):
        super().__init__(env)
        assert 'ansi' in env.metadata["render.modes"]
        self.width = width
        self.height = height

    def render(self, mode='human', **kwargs):
        text = self.env.render(mode='ansi')
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text)
        # noinspection PyTypeChecker
        return np.array(image)
