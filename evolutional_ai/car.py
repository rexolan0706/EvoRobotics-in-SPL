import math
import pygame
import numpy as np

from . import brain_alt as brain
from . import utils

class Car:
    def __init__(self, position, angle, raceTrack):
        self.surface = pygame.Surface((utils.CAR_WIDTH, utils.CAR_LENGTH))
        self.surface.fill(utils.COL_RED)
        self.surface.set_colorkey(utils.COL_BLACK)
        self.surface.set_alpha(120)

        self.steps = 1
        self.quads = [0,0]
        self.driven_dist = 0.001
        self.cells = np.zeros(utils.SIZE)
        self.visited_cells = 0

        self.raceTrack = raceTrack

        self.input_angles = [int(x) for x in utils.CAR_INPUT_ANGLES.split(",")]

        # create the model (random init)
        self.model = brain.Model()

        self.reset(position, angle)

    def setGreen(self):
        self.surface.fill(utils.COL_GREEN)
        self.surface.set_alpha(255)

    def setRed(self):
        self.surface.fill(utils.COL_RED)
        self.surface.set_alpha(120)

    def reset(self, position, angle):
        self.position = [position[0], position[1]]
        self.angle = -angle + 180.0
        self.velocity = 0.5
        self.angle_velocity = 0.0
        self.crashed = False
        self.finished = False
        self.fitness = 0
        self.steps = 1
        self.quads = [0,0]
        self.driven_dist = 0.001
        self.cells = np.zeros(utils.SIZE)
        self.cells[int(self.position[0])][int(self.position[1])] = 1
        self.visited_cells = 1
        self.setRed()

    def check_collision(self):
        # check both front corners for a black pixel below them
        # if any of them has one, this car is crashed
        dx = math.sin(self.angle) * utils.CAR_WIDTH // 2
        dy = math.cos(self.angle) * utils.CAR_LENGTH // 2
        frontLeft = int(self.position[0] - dx), int(self.position[1] - dy)
        frontRight = int(self.position[0] + dx), int(self.position[1] - dy)

        if (frontLeft[0] < 0 or
            frontLeft[0] > utils.SIZE[0] or
            frontLeft[1] < 0 or
            frontLeft[1] > utils.SIZE[1] or
            frontRight[0] < 0 or
            frontRight[0] > utils.SIZE[0] or
            frontRight[1] < 0 or
            frontRight[1] > utils.SIZE[1] or
            self.raceTrack.get_at(frontLeft) == utils.COL_BLACK or
            self.raceTrack.get_at(frontRight) == utils.COL_BLACK):
            self.crashed = True

    def check_win(self):
        dx = math.sin(self.angle) * utils.CAR_WIDTH // 2
        dy = math.cos(self.angle) * utils.CAR_LENGTH // 2
        frontLeft = int(self.position[0] - dx), int(self.position[1] - dy)
        frontRight = int(self.position[0] + dx), int(self.position[1] - dy)

        if self.raceTrack.get_at(frontLeft) == utils.COL_GREEN or self.raceTrack.get_at(frontRight) == utils.COL_GREEN:
            self.finished = True


    def caluclate_light_intensity(self):
        #calculates the sensor measure on a linear decreasing light
        #the light has a radius of 300
        start_int = 900
        x = np.array(self.position)
        y = np.array([utils.LIGHT_X,utils.LIGHT_Y])

        light_intensity = int(np.linalg.norm((x-y)))
        light_intensity = 1.0 - (light_intensity/start_int)
        #print(light_intensity)

        return max(0,light_intensity)



    def get_distance_in_direction(self, angle):
        # angle is relative to self.angle
        # retuns MAX_DIST if not wall is found closer
        for i in range(utils.MAX_DIST):
            x = self.position[0] + float(i) * math.sin(math.radians(-angle + self.angle))
            y = self.position[1] + float(i) * math.cos(math.radians(-angle + self.angle))

            if int(x) < 0 or int(x) >= utils.SIZE[0] or int(y) < 0 or int(y) >= utils.SIZE[1]:
                return float(i) / utils.MAX_DIST # return wall at this point
            if self.raceTrack.get_at((int(x), int(y))) == utils.COL_BLACK:
                return float(i) / utils.MAX_DIST
        return 1.0 # return MAX_DIST / MAX_DIST

    def compute_input_vector(self):
        # returns a vector of 5 values:
        # the distance to the next wall for 5 directions:
        # left, frontLeft, front, frontRight, right
        # if no wall is found, return MAX_DIST

        ret = np.zeros((1,len(self.input_angles)+1), dtype=float)

        for x in range(len(self.input_angles)):
            ret[0,x] = self.get_distance_in_direction(self.input_angles[x])
        ret[0,-1] = self.caluclate_light_intensity()
        return ret
    
    def perform_autonomous_action(self):
        #print(self.compute_input_vector())
        action = self.model.predict(self.compute_input_vector())

        # first control controls speed
        self.velocity = max(0,action[0, 0])
        #if self.velocity < 0.0:
            #self.velocity = 0
        # second action controls angle velocity
        self.angle_velocity = (action[1, 0]) * utils.MAX_TURN_VELOCITY
        #if abs(self.angle_velocity) > 10.0:
        #    self.angle_velocity = self.angle_velocity / abs(self.angle_velocity) * 10.0

    def update(self):
        # Update position
        if self.crashed or self.finished:
            return

        self.steps +=1
        self.check_collision()
        self.check_win()

        if not self.crashed or not self.finished:
            #print(self.angle_velocity)
            self.angle -= self.angle_velocity
            self.angle = self.angle % 360

            old_pos = self.position.copy()
            #print(self.angle_velocity,self.angle)

            self.position[0] += self.velocity * math.sin(math.radians(self.angle))
            self.position[1] += self.velocity * math.cos(math.radians(self.angle))

        self.driven_dist += np.linalg.norm((np.array(old_pos)-np.array(self.position)))
        if self.cells[int(self.position[0])][int(self.position[1])] == 0:
            self.cells[int(self.position[0])][int(self.position[1])] = 1
            self.visited_cells +=1
        
        # Update surface (because of color change)
        self.draw_surface = pygame.transform.rotate(self.surface, self.angle)
        self.rect = self.draw_surface.get_rect()
        self.rect.center = (int(self.position[0]), int(self.position[1]))
        
    def draw(self, window, drawLines=False):
        window.blit(self.draw_surface, self.rect)
        if drawLines:
            inp = self.compute_input_vector()
            for i,angle in enumerate(self.input_angles):
                xp = self.position[0] +inp[0,i] * utils.MAX_DIST * math.sin(math.radians(-angle + self.angle))
                yp = self.position[1] + inp[0,i] * utils.MAX_DIST * math.cos(math.radians(-angle + self.angle))
                pygame.draw.line(window, utils.COL_BLUE, self.position, (int(xp),int(yp)))            