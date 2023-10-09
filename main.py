import pygame
from pygame.locals import *
import numpy as np
import time
import random
import copy
from FasterOwnNN import Layer_Dense

# settings
# simulation screen
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
# whole screen
FSCREEN_WIDTH = 600
FSCREEN_HEIGHT = 800
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
# blob settings
NUMBER_OF_BLOBS = 100
LIFESPAN_BLOB = 100
BLOB_VISION = 10
MUTATION_CHANCE = 1
MUTATION_GREATNESS = 2
# sim settings
SPEED_OF_SIM = 1  # higher == slower
KILL_ON_BORDER = True
BEST_GO_FURTHER = True  # if enabled best of blobs with proceed to next gen
HOW_MANY_BEST = 5
# food settings
FOOD_OCCURENCE = 2_500  # how many frames need to pass in order to create food
NUMBER_OF_FOOD = 1000
FOOD_NUTRITION = 500
FOOD_ONLY_MIN = True  # if true food won't spawn in cycles, but instead it will never fall down below number_of_food//2


# classes and the whole app
class Blob:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True
        self.hp = LIFESPAN_BLOB
        # initialize NN with input length
        self.brain((BLOB_VISION * 2) ** 2)

    def move(self, x_add, y_add):
        self.hp -= 1  # living consumes health

        if self.hp <= 0:
            self.alive = False
        if self.x + x_add <= BLOB_VISION or self.x + x_add >= SCREEN_WIDTH - BLOB_VISION:
            x_add = 0
            if KILL_ON_BORDER:
                self.alive = False
        if self.y + y_add <= BLOB_VISION or self.y + y_add >= SCREEN_HEIGHT - BLOB_VISION:
            y_add = 0
            if KILL_ON_BORDER:
                self.alive = False

        self.x += x_add
        self.y += y_add

    def brain(self, input_len):
        # Neural Network
        self.input_len = input_len
        self.hidden_1 = Layer_Dense(input_len, 10, 'relu', 'hidden1')
        # self.hidden_2 = Layer_Dense(input_len//2, 10, 'relu', 'hidden2')
        self.output_layer = Layer_Dense(10, 2, 'sigmoid', 'output')

    def mutate(self):
        if self.hp > 7_000:
            print(self.hidden_1.print_weights())
            # print(self.hidden_2.print_weights())
            print(self.output_layer.print_weights())
        self.hp = LIFESPAN_BLOB
        self.hidden_1.random_chance_weights_alternation(MUTATION_CHANCE, MUTATION_GREATNESS)
        # self.hidden_2.random_chance_weights_alternation(MUTATION_CHANCE,MUTATION_GREATNESS)
        self.output_layer.random_chance_weights_alternation(MUTATION_CHANCE, MUTATION_GREATNESS)

    def move_with_input(self, input):
        if len(input) != self.input_len:
            raise Exception("Input is different length than brains input")

        self.hidden_1.forward(input)
        # self.hidden_2.forward(self.hidden_1.output)
        self.output_layer.forward(self.hidden_1.output)
        # output should be -1,0,1
        otp = []

        for val in self.output_layer.output:
            if val < 0.33:
                otp.append(-1)
            elif val < 0.66:
                otp.append(0)
            else:
                otp.append(1)
        self.move(otp[0], otp[1])


class App:  # aka world
    def __init__(self):
        # visuals
        self.size = self.width, self.height = FSCREEN_WIDTH, FSCREEN_HEIGHT
        self.text_color = (255, 255, 255)
        self.button_color = (0, 255, 100)
        # rest
        self.gen = 1
        self.screen = None
        self.blobs = []  # board with blobs objects
        self.blob_cords = []
        self.food = []  # board with food cords
        self._running = True
        self._display_surf = None

        self.frame_counter = 0
        self.food_counter = 0

        self.food_screen = []  # initialy blank board sc. width x sc_height
        row = [0 for _ in range(SCREEN_WIDTH)]
        for _ in range(SCREEN_HEIGHT):
            self.food_screen.append(row.copy())

    def on_init(self):
        pygame.init()
        pygame.font.init()
        # screen
        self.screen = pygame.display.set_mode(self.size)
        self._running = True
        self.screen.fill((0, 0, 0))

        # initial spawn
        self.spawn_blobs(NUMBER_OF_BLOBS)
        self.spawn_food(NUMBER_OF_FOOD)

        # set the blob counter
        self.font = pygame.font.Font(None, 36)

    # BLOBS
    def spawn_blobs(self, number_of_blobs):
        self.blob_cords = []
        # cords
        for i in range(number_of_blobs):
            cord_x = random.randint(BLOB_VISION, SCREEN_WIDTH - BLOB_VISION - 1)
            cord_y = random.randint(BLOB_VISION, SCREEN_HEIGHT - BLOB_VISION - 1)
            while (cord_x, cord_y) in self.blob_cords:
                cord_x = random.randint(0, SCREEN_WIDTH)
                cord_y = random.randint(0, SCREEN_HEIGHT)
            self.blob_cords.append((cord_x, cord_y))

            # creation and append
            b = Blob(cord_x, cord_y)
            self.blobs.append(b)
            self.screen.set_at((cord_x, cord_y), RED)

    def spawn_with_sample(self, best_blobs, number_of_blobs):
        new_blobs = []
        self.blob_cords = []
        index = 0
        while len(new_blobs) != number_of_blobs:
            # find cords for the blob
            cord_x = random.randint(BLOB_VISION, SCREEN_WIDTH - BLOB_VISION - 1)
            cord_y = random.randint(BLOB_VISION, SCREEN_HEIGHT - BLOB_VISION - 1)
            while (cord_x, cord_y) in self.blob_cords:
                cord_x = random.randint(0, SCREEN_WIDTH)
                cord_y = random.randint(0, SCREEN_HEIGHT)
            self.blob_cords.append((cord_x, cord_y))

            new_blob = copy.deepcopy(best_blobs[index])
            new_blob.mutate()
            new_blob.x = cord_x
            new_blob.y = cord_y
            new_blobs.append(new_blob)
            index += 1
            if index == len(best_blobs):
                index = 0
        self.blobs = new_blobs

    def check_blobs_alive(self):
        for b in self.blobs:
            if not b.alive:
                self.blobs.remove(b)

    def get_blob_vision(self, x, y):
        # create input (20 x 20 pixels around blob), function returns flattened version
        # if x - BLOB_VISION < 0:
        #     x = BLOB_VISION
        # elif x + BLOB_VISION > SCREEN_WIDTH:
        #     x = BLOB_VISION * -1
        # if y - BLOB_VISION < 0:
        #     y = BLOB_VISION
        # elif y + BLOB_VISION > SCREEN_HEIGHT:
        #     y = BLOB_VISION * -1

        surr = np.array(self.food_screen[x - BLOB_VISION:x + BLOB_VISION])
        surr = surr[:, y - BLOB_VISION:y + BLOB_VISION]
        vision = list(surr.flatten())
        return vision

    def random_move_blobs(self):
        moves = [-1, 1, 0]
        for blob in self.blobs:
            blob.move(random.choice(moves), random.choice(moves))

    def move_blobs(self):
        for blob in self.blobs:
            blob.move_with_input(self.get_blob_vision(blob.x, blob.y))

    def blob_reproduce(self, blob):
        blob_clone = copy.deepcopy(blob)
        blob_clone.mutate()
        self.blobs.append(blob_clone)

    # FOOD
    def spawn_food(self, n):  # spawns n unique placed food
        for i in range(n):
            cord_x = random.randint(SCREEN_WIDTH // 4, SCREEN_WIDTH // 4 * 3 - 1)
            cord_y = random.randint(SCREEN_HEIGHT // 4, SCREEN_HEIGHT // 4 * 3 - 1)
            while (cord_x, cord_y) in self.blob_cords or (cord_x, cord_y) in self.food:
                cord_x = random.randint(SCREEN_WIDTH // 4, SCREEN_WIDTH // 4 * 3 - 1)
                cord_y = random.randint(SCREEN_HEIGHT // 4, SCREEN_HEIGHT // 4 * 3 - 1)

            self.food.append((cord_x, cord_y))
            self.screen.set_at((cord_x, cord_y), GREEN)
            self.food_screen[cord_x][cord_y] = 1  # update food_screen

    def feed_blobs(self):
        for blob in self.blobs:
            if (blob.x, blob.y) in self.food:
                blob.hp += FOOD_NUTRITION
                self.food.remove((blob.x, blob.y))

                # reproduce
                self.blob_reproduce(blob)

    # rendering and simulation
    def render_all(self):
        self.screen.fill((0, 0, 0))  # reset screen

        for b in self.blobs:  # add blobs
            self.screen.set_at((b.x, b.y), RED)
        for cords in self.food:
            self.screen.set_at(cords, GREEN)

        # blob counter
        self.score_text = self.font.render(f'Blobs:{len(self.blobs)}', True, self.text_color)
        self.screen.blit(self.score_text, (10, 10))
        # gen counter
        self.gen_text = self.font.render(f'Gen:{self.gen}', True, self.text_color)
        self.screen.blit(self.gen_text, (10, 30))
        # border
        pygame.draw.rect(self.screen, (150, 150, 150), (0, SCREEN_HEIGHT, FSCREEN_WIDTH, 5))
        # buttons
        # Mutate +
        pygame.draw.rect(self.screen, self.button_color, (0, SCREEN_HEIGHT + 5, 200, 50))
        text = self.font.render("Mutate+", True, self.text_color)
        text_rect = text.get_rect(center=(100, SCREEN_HEIGHT + 30))
        self.screen.blit(text, text_rect)
        # Mutate -
        pygame.draw.rect(self.screen, self.button_color, (200, SCREEN_HEIGHT + 5, 200, 50))
        text = self.font.render("Mutate-", True, self.text_color)
        text_rect = text.get_rect(center=(300, SCREEN_HEIGHT + 30))
        self.screen.blit(text, text_rect)

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            global MUTATION_GREATNESS
            # Sprawdzenie, czy kliknięcie myszy było w obszarze guzika
            if 0 <= event.pos[0] <= 200 and SCREEN_HEIGHT + 5 <= event.pos[1] <= SCREEN_HEIGHT + 55:
                print("Mutate+")
                MUTATION_GREATNESS += 1
                print(f"Mutation:{MUTATION_GREATNESS}")
            elif 200 <= event.pos[0] <= 400 and SCREEN_HEIGHT + 5 <= event.pos[1] <= SCREEN_HEIGHT + 55:
                print("Mutate-")
                MUTATION_GREATNESS -= 1
                print(f"Mutation:{MUTATION_GREATNESS}")

    def on_loop(self):  # main loop of the app

        if self.frame_counter == SPEED_OF_SIM:
            self.check_blobs_alive()  # checking living blobs
            if len(self.blobs) <= HOW_MANY_BEST and BEST_GO_FURTHER:
                self.gen += 1
                self.spawn_with_sample(self.blobs, NUMBER_OF_BLOBS)
                print(len(self.blobs))
            elif len(self.blobs) == 0 and not BEST_GO_FURTHER:
                self.gen += 1
                self.spawn_blobs(NUMBER_OF_BLOBS)
            # self.random_move_blobs()  # random move
            self.move_blobs()
            self.feed_blobs()  # check if blobs are on food, known bug: 2 blobs on the same food
            self.frame_counter = 0

            # food checking
            if FOOD_ONLY_MIN and len(self.food) <= NUMBER_OF_FOOD // 2:
                self.spawn_food(NUMBER_OF_FOOD)
            elif not FOOD_ONLY_MIN:
                if self.food_counter == FOOD_OCCURENCE:
                    self.food_counter = 0
                    self.spawn_food(NUMBER_OF_FOOD)  # spawning food
                self.food_counter += 1

            # render
            self.render_all()
            pygame.display.flip()

        self.frame_counter += 1

    def on_render(self):
        pass

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
