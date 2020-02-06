import numpy as np
import math
import pygame


class Person(object):

    def __init__(self, amount):
        self.health = 200
        self.coordinates = []
        self.amount_of_food = amount
        # the input is the shortest distance to an apple and the starting value doesnt matter as long as it is really high
        # [current shortest distance to an apple, last shortest distance to an apple]
        self.input = [10000, 10000]
        # the weights for each layer
        self.layer_two = []
        self.layer_one = []
        # this is the four directions that the person can move in
        self.outputs = []
        self.speed = 0
        self.bias_one = 0
        self.bias_two = 0
        self.radius = 10
        self.angle = 0
        self.delta_angle = 0
        self.last_known_locations = []

    def initalize_person(self):
        # creates random weights for the network
        self.layer_two = np.random.rand(6,2)
        self.layer_one = np.random.rand(2,6)
        # everyone has a different starting point [x, y]
        self.coordinates = [np.random.randint(50, 750), np.random.randint(50, 550)]
        # everyone starts with a random set of outputs
        self.outputs = [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]
        # everyone starts with a random speed and bias
        self.speed = np.random.randint(1, 5)
        # everyone starts with two random biases for the network
        self.bias_one = np.random.rand()
        self.bias_two = np.random.rand()
        # they all start off facing a random direction
        self.angle = np.random.rand()
        self.delta_angle = np.random.randint(1, 11)

    def get_input(self, food):
        # this determines the distance between the person and each apple
        x_distance = (self.coordinates[0] - food.x) ** (2)  
        y_distance = (self.coordinates[1] - food.y) ** (2)
        total_distance = (x_distance + y_distance) ** (1/2)
        if self.input[0] > total_distance:
            self.input[0] = total_distance

    def softmax(self, x):
        # this is the activation function for the neural network
        return np.exp(x) / np.sum(np.exp(x))
        #return 1/(1 + np.exp(-x))

    def nueral_network(self):
        # this is a simple feedforward neural network
        feed_forward_one = np.dot(self.input, self.layer_one) + self.bias_one
        layer = self.softmax(feed_forward_one)
        feed_forward_two = np.dot(layer, self.layer_two) + self.bias_two
        self.outputs = self.softmax(feed_forward_two)
        #print(self.outputs)

    def move_person(self, screen):
        if self.outputs[0] > self.outputs[1]:
            self.angle += 10
        else:
            self.angle -= 10
        dy = math.sin(math.radians(self.angle)) * self.speed
        dx = math.cos(math.radians(self.angle)) * self.speed
        self.coordinates[0] += dx
        self.coordinates[1] += dy
        #if self.radius >= 5:
        self.person = pygame.draw.circle(screen, (0, 0, 255), (int(self.coordinates[0]), int(self.coordinates[1])), self.radius)


class Food(object):

   def __init__(self, screen):
       self.x = 0
       self.y = 0
       self.screen = screen

   def create_new_food(self):
       # puts the food in a different location
       self.x = np.random.randint(50, 750)
       self.y = np.random.randint(50, 550)

   def draw_food(self):
       # draws the food on top of the background
       apple = pygame.image.load(r'apple.png')
       self.screen.blit(apple, (self.x, self.y))


class Environment(object):

    def __init__(self):
        self.initial_pop_size = 100
        self.amount_of_food = 20
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.running = True
        self.generation_count = 0
        self.count = 0
        self.population = []
        self.food = []
        self.index_for_deletion = []
        self.chance_for_selection = 0.5
        self.chance_for_mutation = 0.05
 
    # this is where the background is set up
    def draw_environment(self):
        background = pygame.image.load(r'trees.jpg')
        self.screen.blit(background, (0, 0))

    def selection(self):
        if len(self.population) > 1:
            one, two = np.random.randint(0, len(self.population)), np.random.randint(0, len(self.population))
            person_one = self.population[one]
            person_two = self.population[two]
            people = [person_one, person_two]
            return people
        return None

    def crossover(self, people):
        if people is not None:
            one = people[0]
            two = people[1]
            child_layer_one = [[], []]
            child_layer_two = [[], [], [], [], [], []]
            for i in range(6):
                for j in range(2):
                    chance = np.random.rand()
                    if chance <= 0.5:
                        child_layer_one[j].append(one.layer_one[j][i])
                    else:
                        child_layer_one[j].append(two.layer_one[j][i])
                    second_chance = np.random.rand()
                    if second_chance <= 0.5:
                        child_layer_two[i].append(one.layer_two[i][j])
                    else:
                        child_layer_two[i].append(two.layer_two[i][j])
            child = Person(self.amount_of_food)
            child.initalize_person()
            child.layer_one = child_layer_one
            child.layer_two = child_layer_two
            child.speed = one.speed
            child.bias_one = one.bias_one
            child.bias_two = two.bias_two
            child.delta_angle = one.delta_angle
            return child
        return None

    def mutation(self, child):
        if child is not None:
            for i in range(6):
                mutation = np.random.rand()
                for j in range(2):
                    mutation_two = np.random.rand()
                    if self.chance_for_mutation >= mutation:
                        child.layer_one[j][i] = np.random.rand()
                    if self.chance_for_mutation >= mutation_two:
                        child.layer_two[i][j] = np.random.rand()
            self.population.append(child)

    # this is the main body of code
    def main(self):
        # starts pygame
        pygame.init()
        # runs before the loop containing the algorithm
        for i in range(self.initial_pop_size):
            person = Person(self.amount_of_food)
            person.initalize_person()
            self.population.append(person)
        for i in range(self.amount_of_food):
            food = Food(self.screen)
            food.create_new_food()
            self.food.append(food)
        # starts the loop that runs the algorithm
        while self.running:
            if self.count % 200 == 0:
                self.generation_count += 1
                print(self.generation_count)
            self.count += 1
            # if the X is clicked then stop the loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            # updates the display
            pygame.display.update()
            self.draw_environment()
            # draws all the food
            for i in range(self.amount_of_food):
                self.food[i].draw_food()
            # goes through the population
            for i in range(len(self.population)):
                person = self.population[i]
                person.input[0] = 10000
                for j in range(self.amount_of_food):
                    food = self.food[j]
                    person.get_input(food)
                    # checks if food has been eaten
                    if person.coordinates[0] <= food.x+10 and person.coordinates[0] >= food.x and person.coordinates[1] <= food.y+10 and person.coordinates[1]>= food.y:
                    #if person.coordinates[0] == food.x and person.coordinates[1] == food.y:
                        person.health += 400
                        food.create_new_food()
                        person.radius += 2
                # draws the person
                person.nueral_network()
                person.move_person(self.screen)
                # forces the person to eat to survive
                person.health -= 1
                # doesnt let the person go out of bounds
                if person.coordinates[0] > 800 or person.coordinates[0] < 0 or person.coordinates[1] > 600 or person.coordinates[1] < 0:
                    person.health -= 10000
                # changes the size of the person as they get skinnier
                if self.count % 50 == 0 and person.radius > 0:
                    person.radius -= 1
                # determines if the individual should die or not
                if person.health <= 0:
                    self.index_for_deletion.append(person)
                person.input[1] = person.input[0]
                # stops them from just going around in circles
                if len(person.last_known_locations) >= 100:
                    person.last_known_locations.pop()
                for j in range(len(person.last_known_locations)):
                    if person.input[0] == person.last_known_locations[j]:
                        person.health -= 10000
                person.last_known_locations.append(person.input[0])
            for i in self.index_for_deletion:
                self.population.remove(i)
            del self.index_for_deletion[:]
            # the steps in the algorithm (the fitness function is already taken care of
            # this runs all the time so the longer a person is alive for the better the chances of them being selected
            selection = np.random.rand()
            if len(self.population) >= 100:
                selection = 1
            if self.chance_for_selection >= selection:
                select = self.selection()
                child = self.crossover(select)
                self.mutation(child)


# runs the program
env = Environment()
env.main()

# the two inputs could be, A) the closest distance to an apple they've every seen, and B) how close they currently are to the apple closest to them
