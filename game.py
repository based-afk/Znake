import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.panel_width = 340
        self.display = pygame.display.set_mode((self.w + self.panel_width, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.last_q_values = [0.0, 0.0, 0.0]
        self.frame_history = []
        self.step_mode = False
        self.waiting_for_step = False
        self.show_panel = False


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.step_mode = not self.step_mode
                if event.key == pygame.K_h:
                    self.show_panel = not self.show_panel
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        # --- Danger zone highlights ---
        if self.last_state is not None and self.last_action is not None:
            s = self.last_state
            head = self.snake[0]

            dir_r = self.direction == Direction.RIGHT
            dir_l = self.direction == Direction.LEFT
            dir_u = self.direction == Direction.UP
            dir_d = self.direction == Direction.DOWN

            if dir_r:
                straight = Point(head.x + BLOCK_SIZE, head.y)
                right = Point(head.x, head.y + BLOCK_SIZE)
                left = Point(head.x, head.y - BLOCK_SIZE)
            elif dir_l:
                straight = Point(head.x - BLOCK_SIZE, head.y)
                right = Point(head.x, head.y - BLOCK_SIZE)
                left = Point(head.x, head.y + BLOCK_SIZE)
            elif dir_u:
                straight = Point(head.x, head.y - BLOCK_SIZE)
                right = Point(head.x + BLOCK_SIZE, head.y)
                left = Point(head.x - BLOCK_SIZE, head.y)
            else:
                straight = Point(head.x, head.y + BLOCK_SIZE)
                right = Point(head.x - BLOCK_SIZE, head.y)
                left = Point(head.x + BLOCK_SIZE, head.y)

            danger_cells = [
                (straight, s[0]),
                (right, s[1]),
                (left, s[2]),
            ]

            overlay = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            for pt, is_danger in danger_cells:
                if 0 <= pt.x < self.w and 0 <= pt.y < self.h:
                    if is_danger:
                        overlay.fill((255, 0, 0, 120))
                    else:
                        overlay.fill((0, 255, 0, 60))
                    self.display.blit(overlay, (pt.x, pt.y))

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        if self.show_panel:
            self._draw_panel()
        pygame.display.flip()

    def _draw_panel(self):
        panel_x = self.w + 10
        small_font = pygame.font.SysFont('arial', 14)
        med_font = pygame.font.SysFont('arial', 16, bold=True)

        def label(text, y, color=(200, 200, 200)):
            surf = small_font.render(text, True, color)
            self.display.blit(surf, (panel_x, y))

        def header(text, y, color=(255, 220, 50)):
            surf = med_font.render(text, True, color)
            self.display.blit(surf, (panel_x, y))

        header(f"Frame {self.frame_iteration}", 10)
        mode_text = "[ STEP MODE ]" if self.step_mode else "[ PLAY ]  (P to pause)"
        label(mode_text, 32, (100, 255, 100) if self.step_mode else (150, 150, 150))

        header("STATE VECTOR", 65)
        if self.last_state is not None:
            s = self.last_state
            entries = [
                ("Danger straight", s[0]),
                ("Danger right",    s[1]),
                ("Danger left",     s[2]),
                ("Moving left",     s[3]),
                ("Moving right",    s[4]),
                ("Moving up",       s[5]),
                ("Moving down",     s[6]),
                ("Food left",       s[7]),
                ("Food right",      s[8]),
                ("Food up",         s[9]),
                ("Food down",       s[10]),
            ]
            for i, (lbl, val) in enumerate(entries):
                color = (255, 80, 80) if (val and i < 3) else \
                        (80, 255, 80) if val else (130, 130, 130)
                icon = "! " if (val and i < 3) else ("> " if val else "  ")
                label(f"{icon}{lbl}: {int(val)}", 85 + i * 18, color)

        header("Q-VALUES", 295)
        action_names = ["Straight", "Turn Right", "Turn Left"]
        q = self.last_q_values
        bar_max = max(abs(v) for v in q) if any(q) else 1
        for i, (name, val) in enumerate(zip(action_names, q)):
            y = 315 + i * 38
            chosen = (self.last_action is not None and self.last_action[i] == 1)
            bar_color = (50, 200, 50) if chosen else (80, 120, 200)
            bar_len = int(abs(val) / (bar_max + 1e-9) * 140)
            pygame.draw.rect(self.display, (40, 40, 40), (panel_x, y + 16, 140, 14))
            pygame.draw.rect(self.display, bar_color, (panel_x, y + 16, bar_len, 14))
            tag = " < CHOSEN" if chosen else ""
            label(f"{name}: {val:.2f}{tag}", y, (255, 255, 100) if chosen else (180, 180, 180))

        header("MEMORY", 435)
        label(f"Last reward: {self.last_reward:+.0f}", 455,
              (50, 255, 50) if self.last_reward > 0 else (255, 80, 80) if self.last_reward < 0 else (180, 180, 180))

        header("CONTROLS", 490)
        label("P       - toggle step mode", 510)
        label("SPACE   - next frame (step mode)", 528)
        label("H       - toggle panel", 546)


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)