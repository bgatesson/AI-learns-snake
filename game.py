import pygame
import numpy as np

pygame.init()
font = pygame.font.SysFont("arial", 20)

# colors
green = (0,255,0)
red = (255,0,0)
black = (0,0,0)
white = (255,255,255)

block_size = 25 
snake_speed = 10

class SnakeGameAI:

    def __init__(self, w=800, h=600):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w ,self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock() 
        self.reset()
    
    def reset(self):
        # init game state 
        self.direction = 'RIGHT'

        self.snakehead = (self.w/2, self.h/2)
        self.snakebody = [self.snakehead]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iter = 0

    def play(self, action):
        self.frame_iter += 1
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # snake movement
        self._move(action)
        self.snakebody.insert(0, self.snakehead)

        # check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iter > (100*len(self.snakebody)):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # place new food and update score // update body position if just moving
        if self.snakehead == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snakebody.pop()

        # update screen and clock
        self._update_ui()
        self.clock.tick(snake_speed)

        # return game over and user's score
        return reward, game_over, self.score
    
    def _move(self, action):
        # [straight, right turn, left turn]
        clockwise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # straight
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]): # right turn
            new_dir = clockwise[((idx + 1) % 4)]
        elif np.array_equal(action, [0, 0, 1]): # left turn
            new_dir = clockwise[((idx - 1) % 4)]

        self.direction = new_dir

        x = self.snakehead[0]
        y = self.snakehead[1]
        if self.direction == 'RIGHT':
            x += block_size
        elif self.direction == 'LEFT':
            x -= block_size
        elif self.direction == 'UP':
            y -= block_size
        elif self.direction == 'DOWN':
            y += block_size
        
        self.snakehead = (x, y)

    def _update_ui(self):
        self.display.fill(black)

        # draw snake body
        for part in self.snakebody:
            pygame.draw.rect(self.display, green, (part[0], part[1], block_size, block_size))
        # draw food
        pygame.draw.rect(self.display, red, (self.food[0], self.food[1], block_size, block_size))

        text = font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def is_collision(self, part=None):
        if part is None:
            part = self.snakehead
        # snake hits boundaries
        if part[0] >= self.w or part[0] < 0 or part[1] >= self.h or part[1] < 0:
            return True
        # snake hits a body part
        if part in self.snakebody[1:]:
            return True
        
        return False
    
    def _place_food(self):
        x = np.random.randint(0, (self.w-block_size)//block_size)*block_size
        y = np.random.randint(0, (self.h-block_size)//block_size)*block_size
        self.food = (x, y)
        if self.food in self.snakebody:
            self._place_food()



