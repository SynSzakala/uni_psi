from dataclasses import dataclass
from enum import IntEnum
from math import sqrt
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class PositionDelta:
    dx: int
    dy: int


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @classmethod
    def delta(cls, obj: int) -> PositionDelta:
        return [PositionDelta(0, -1), PositionDelta(1, 0), PositionDelta(0, 1), PositionDelta(-1, 0)][obj]


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    @staticmethod
    def from_iterable(coordinates: Iterable[int]):
        x, y = coordinates
        return Position(x, y)

    def __add__(self, other: PositionDelta):
        return Position(self.x + other.dx, self.y + other.dy)

    def move(self, direction: Direction):
        return self + Direction.delta(direction)

    @staticmethod
    def distance(a, b) -> float:
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def random_position(self):
        return Position(np.random.randint(0, self.width), np.random.randint(0, self.height))

    def contains_position(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return Position(self.width // 2, self.height // 2)


class TileType(IntEnum):
    NONE = 0
    FOOD = 1
    SNAKE_HEAD = 2
    SNAKE_BODY = 3

    @classmethod
    def str(cls, obj) -> str:
        str_map = {TileType.NONE: ' ', TileType.FOOD: '*', TileType.SNAKE_HEAD: '0', TileType.SNAKE_BODY: 'x'}
        return str_map[obj]


@dataclass
class SnakeState:
    size: Size
    head: Position
    body: List[Position]
    pending_body_extend: bool
    direction: Direction
    food: List[Position]

    @staticmethod
    def generate_initial(size: Size):
        return SnakeState(
            size=size,
            head=size.center,
            body=[],
            pending_body_extend=False,
            direction=np.random.choice(Direction),
            food=[],
        )

    def get_tile(self, position: Position) -> TileType:
        if position == self.head:
            return TileType.SNAKE_HEAD
        if position in self.food:
            return TileType.FOOD
        return TileType.NONE

    def build_tiles(self) -> np.ndarray:
        tiles = np.full((len(TileType), self.size.height, self.size.width), 0)
        tiles[TileType.NONE, :] = 1
        SnakeState.__set_tiles(tiles, [self.head], TileType.SNAKE_HEAD)
        SnakeState.__set_tiles(tiles, self.body, TileType.SNAKE_BODY)
        SnakeState.__set_tiles(tiles, self.food, TileType.FOOD)
        return tiles

    def can_move(self, position: Position) -> bool:
        return self.size.contains_position(position) and position not in self.body

    def is_empty(self, position: Position) -> bool:
        return position != self.head and position not in self.food and position not in self.body

    def __str__(self) -> str:
        content = ""
        tiles = self.build_tiles()
        for y in range(self.size.height):
            if y > 0:
                content += '\n'
            for x in range(self.size.width):
                if x > 0:
                    content += ' '
                content += TileType.str(np.where(tiles[:, y, x] == 1)[0].item(0))
        return bordered_text(content)

    @staticmethod
    def __set_tiles(tiles: np.ndarray, positions: Iterable[Position], tile_type: TileType):
        for position in positions:
            tiles[:, position.y, position.x] = 0
            tiles[tile_type, position.y, position.x] = 1


def bordered_text(text: str) -> str:
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['*' + '-' * width + '*']
    for s in lines:
        res.append('|' + (s + ' ' * width)[:width] + '|')
    res.append('*' + '-' * width + '*')
    return '\n'.join(res)
