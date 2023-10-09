import pygame
import random
from DeepQNN import DeepQN
import numpy as np

# app params
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
MENU_HEIGHT = 50
FULL_HEIGHT = SCREEN_HEIGHT + MENU_HEIGHT
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (0, 255, 100)
# sim settings
BLOB_LIFESPAN = 5000
BLOB_VISION = 10
NUMBER_OF_FOOD = 1000
STEPS_IN_LEARNING = 100
# NN settings
GAMMA = 0.995  # discount factor

# rewards
FOOD_REWARD = 100
MOVE_REWARD = -1


# REPRODUCTION_REWARD = 100

class Blob:
    def __init__(self, x, y, lifespan):
        self.x = x
        self.y = y
        self.lifespan = lifespan

    def move(self, lr, ud):
        if (self.x == SCREEN_WIDTH - BLOB_VISION and lr >= 1) or (self.x == BLOB_VISION and lr <= -1):
            pass
        else:
            self.x += lr

        if (self.y == SCREEN_HEIGHT - BLOB_VISION and ud >= 1) or (self.y == BLOB_VISION and ud <= -1):
            pass
        else:
            self.y += ud

        self.lifespan -= 1


class App:
    def __init__(self):
        pygame.init()
        self.running = True
        self.size = self.width, self.height = SCREEN_WIDTH, FULL_HEIGHT
        self.move_history = [[], [], [], []]
        # blob settings
        self.blob = Blob(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BLOB_LIFESPAN)

        # food settings
        self.food = []  # board with food cords
        self.food_screen = []  # initialy blank board sc. width x sc_height
        row = [0 for _ in range(SCREEN_WIDTH)]
        for _ in range(SCREEN_HEIGHT):
            self.food_screen.append(row.copy())
        self.spawn_food(NUMBER_OF_FOOD)

        # Deep Q Network
        state_size = ((BLOB_VISION * 2) ** 2,)
        self.NN = DeepQN(state_size, 4, 0.01)

        # rest
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.Font(None, 36)
        pygame.display.set_caption("Blobz")

    # blob

    def get_blob_vision(self):
        # return vision (2*BLOB_VISION x 2*BLOB_VISION pixels around blob, flattened)
        x, y = self.blob.x, self.blob.y
        if x - BLOB_VISION < 0:
            x = BLOB_VISION
        elif x + BLOB_VISION > SCREEN_WIDTH:
            x = SCREEN_WIDTH - BLOB_VISION - 1
        if y - BLOB_VISION < 0:
            y = BLOB_VISION
        elif y + BLOB_VISION > SCREEN_HEIGHT:
            y = SCREEN_HEIGHT - BLOB_VISION - 1

        surr = np.array(self.food_screen[x - BLOB_VISION:x + BLOB_VISION])
        surr = surr[:, y - BLOB_VISION:y + BLOB_VISION]
        vision = surr.flatten()
        return vision

    def single_step(self):
        epsilon = random.random()
        state = self.get_blob_vision().reshape(1, -1)
        reward = self.reward(self.blob.x, self.blob.y)

        if epsilon < 0.95:
            actions = self.NN.forward(state)
            best_action = np.argmax(actions)  # 0,1,2,3
        else:
            best_action = random.randint(0, 4)

        if best_action == 0:  # right
            self.blob.move(1, 0)
        elif best_action == 1:  # left
            self.blob.move(-1, 0)
        elif best_action == 2:  # up
            self.blob.move(0, 1)
        elif best_action == 3:  # down
            self.blob.move(0, -1)

        next_state = self.get_blob_vision().reshape(1, -1)
        self.move_history[0].append(state)
        self.move_history[1].append(best_action)
        self.move_history[2].append(reward)
        self.move_history[3].append(next_state)

    def reward(self, blob_x, blob_y):
        blob_cords = (blob_x, blob_y)
        if blob_cords in self.food:
            self.food.remove(blob_cords)
            return FOOD_REWARD
        else:
            return MOVE_REWARD

    def learn(self, steps):
        for _ in range(10):
            self.move_history = [[], [], [], []]

            for i in range(steps):
                self.single_step()
                self.render_all()
                pygame.display.update()
            self.NN.learn(self.move_history, GAMMA)

    def spawn_food(self, n):  # spawns n unique placed food
        for i in range(n):
            cord_x = random.randint(0, SCREEN_WIDTH - 1)
            cord_y = random.randint(0, SCREEN_HEIGHT - 1)
            while (cord_x, cord_y) == (self.blob.x, self.blob.y) or (cord_x, cord_y) in self.food:
                cord_x = random.randint(0, SCREEN_WIDTH - 1)
                cord_y = random.randint(0, SCREEN_HEIGHT - 1)

            self.food.append((cord_x, cord_y))
            self.food_screen[cord_x][cord_y] = 1  # update food_screen

    def render_all(self):  # render screen of simulation
        self.screen.fill((0, 0, 0))  # reset screen
        # blob
        self.screen.set_at((self.blob.x, self.blob.y), RED)

        # food
        for cords in self.food:
            self.screen.set_at(cords, GREEN)

        # buttons
        pygame.draw.rect(self.screen, BUTTON_COLOR, (0, FULL_HEIGHT - 50, 100, 50))
        text = self.font.render("Learn", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(50, FULL_HEIGHT - 25))
        self.screen.blit(text, text_rect)

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if 0 <= event.pos[0] <= 100 and FULL_HEIGHT - 50 <= event.pos[1] <= FULL_HEIGHT:
                self.learn(STEPS_IN_LEARNING)

    def main_loop(self):
        while self.running:
            for event in pygame.event.get():
                self.on_event(event)
            # main loop
            self.render_all()
            pygame.display.update()

        pygame.quit()


if __name__ == "__main__":
    app = App()
    app.main_loop()
