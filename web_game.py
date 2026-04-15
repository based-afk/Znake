import os
from enum import Enum
from collections import namedtuple
from io import BytesIO

import numpy as np
import pygame
from PIL import Image

# Ensure pygame uses a headless video driver when running on a server.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame.init()
pygame.font.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
BASE_SPEED = 8
AI_SPEED = 40


class WebSnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.surface = pygame.Surface((self.w, self.h))
        self.font = pygame.font.Font("arial.ttf", 25)
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.speed = BASE_SPEED
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.last_q_values = [0.0, 0.0, 0.0]

    def _place_food(self):
        x = np.random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE + 1) * BLOCK_SIZE
        y = np.random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE + 1) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def step_human(self, new_direction=None):
        if new_direction is not None:
            self.direction = new_direction

        self._move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self.speed = BASE_SPEED + self.score
            self._place_food()
        else:
            self.snake.pop()

        return game_over, self.score

    def step_ai(self, action):
        self.frame_iteration += 1
        self._move_ai(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _move_ai(self, action):
        # action: [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        self._move(self.direction)

    def _render(self):
        self.surface.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.surface, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.surface, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.surface, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.surface.blit(text, [0, 0])

    def get_frame_bytes(self):
        self._render()
        frame = pygame.surfarray.array3d(self.surface)
        frame = np.transpose(frame, (1, 0, 2))
        image = Image.fromarray(frame)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
