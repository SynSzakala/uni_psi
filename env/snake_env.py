import numpy as np
from gym import Env, spaces as spaces

from env.snake_utils import Direction, SnakeState, Size, TileType


class SnakeEnv(Env):
    metadata = {'render.modes': ['human', 'ansi']}
    action_space = spaces.Discrete(len(Direction))

    state: SnakeState

    def __init__(self, size: Size, food_probability: float = 0.1):
        self.size = size
        self.food_probability = food_probability
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(TileType), size.height, size.width),
            dtype=np.uint8,
        )

    def step(self, action: Direction | None):
        if action is not None:
            self.state.direction = action
        done = self.__maybe_move_snake()
        if done:
            reward = -10
        else:
            if self.__maybe_consume_food():
                reward = 1
                done = len(self.state.body) > self.size.area * 0.5
            else:
                reward = -0.01
            self.__maybe_spawn_food()

        return self.state.build_tiles(), reward, done, {}

    def render(self, mode='human'):
        content = self.state.__str__()
        if mode == 'human':
            print(content)
        if mode == 'ansi':
            return content

    def reset(self):
        self.state = SnakeState.generate_initial(self.size)
        return self.state.build_tiles()

    def __maybe_spawn_food(self):
        if np.random.random() <= self.food_probability:
            while True:
                position = self.size.random_position()
                if self.state.is_empty(position):
                    self.state.food.append(position)
                    return

    def __maybe_consume_food(self) -> bool:
        if self.state.head in self.state.food:
            self.state.food.remove(self.state.head)
            self.state.pending_body_extend = True
            return True
        else:
            return False

    def __maybe_move_snake(self) -> bool:
        new_head = self.state.head.move(self.state.direction)
        if self.state.can_move(new_head):
            self.state.body.insert(0, self.state.head)
            self.state.head = new_head
            if not self.state.pending_body_extend:
                self.state.body.pop()
            else:
                self.state.pending_body_extend = False
            return False
        else:
            return True
