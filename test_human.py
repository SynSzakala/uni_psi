from gym.utils.play import play

from env.snake_utils import Direction, Size
from env.snake_env import SnakeEnv
from text_to_image_env import TextToImageEnv

env = SnakeEnv(size=Size(10, 10))

wrapped_env = TextToImageEnv(env, width=300, height=300)

play(
    wrapped_env,
    keys_to_action={
        (): None,
        (ord('w'),): Direction.UP,
        (ord('s'),): Direction.DOWN,
        (ord('a'),): Direction.LEFT,
        (ord('d'),): Direction.RIGHT
    },
    fps=2
)
