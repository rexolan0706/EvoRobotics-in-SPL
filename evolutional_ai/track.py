import os
import pygame
import numpy as np
import math

from . import utils

class Track:
    def __init__(self, filename):
        self.filename = filename

        self.image = None
        self.data = []

        self.thickness = utils.THICKNESS

    def set_data(self, data):
        self.data = data
    
    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_image(self, image):
        self.image = image

    def add_point(self, x, y):
        # adds a point to the track
        if len(self.data) > 0:
            # prevent twice the same point
            if self.data[-1][0] == x and self.data[-1][1] == y:
                return
        self.data.append((x,y))

    def remove_point(self):
        if len(self.data) > 0:
            self.data.pop()

    def load(self):
        self.image = pygame.image.load(os.path.join(utils.TRACKS_PATH, self.filename + ".png"))
        if utils.TRAIN_STEPS == 1:
            points = [[250,250],[300,300],[450,450],[250,450],[450,250],]
            self.data = [points[np.random.randint(0,len(points))]]
        elif not utils.TRAIN_STEPS == 3:
            self.data =[[np.random.randint(50,utils.SIZE[0]-50),np.random.randint(50,utils.SIZE[1]-50)]]
        else:
            self.data = np.loadtxt(os.path.join(utils.TRACKS_PATH, self.filename + ".txt"))


        # angle for first line
        #if(len(self.data) < 2):
            #print("Error: Track not valid.")
            #quit(1)

        if not utils.TRAIN_STEPS == 3:
            self.init_angle = np.random.randint(0,360)
        else:
            dx = self.data[1][0] - self.data[0][0]
            dy = self.data[1][1] - self.data[0][1]
            self.init_angle = -int(math.degrees(math.atan(-dy/dx))) + 90
            #self.init_angle = 90

    def get_start_info(self):
        if self.image:
            return self.data[0], self.init_angle
        else:
            print("Error: Not yet loaded")
            return None, None

    def save(self):
        pygame.image.save(self.image, os.path.join(utils.TRACKS_PATH, self.filename + ".png"))
        np.savetxt(os.path.join(utils.TRACKS_PATH, self.filename + ".txt"), self.data)
        # vectoresh save mishe

    def get_length(self):
        if len(self.data) > 2:
            return 0.0
        else:
            ret = 0.0
            sx,sy = self.data[0]
            for cx,cy in self.data[1:]:
                ret += math.sqrt((sx-cx)**2 + (sy-cy)**2)
                sx,sy = cx,cy
            return ret
        
    
    def calc_quadrant(self,x,y,quads):
        if x < 300:
            quads[0]+=1

        if y < 300:
            quads[1]+=1
    
        return quads

        
    def get_fitness(self,x,y,quads,steps,inp_vec,driven_dist,cells_visited):
        #calculate the fitness during the simulation based on the chosen behaviour

        train_step = utils.TRAIN_STEPS

        #calculate the fitness for homing during the simulation
        # this is just based on the euclidean distance to the goal
        if train_step == 0:
            #exploration
            max_steps = (600*600)
            return (max_steps-cells_visited)
        
        elif train_step == 1:
            #homing
            a = np.array((x,y))
            b = np.array((75,75))
            fitness = 0.1 * np.linalg.norm(a-b)
            return fitness
        
        elif train_step == 2:
            #obstacle avoidance
            max_steps = (600*600)
            sens_dist = np.mean(inp_vec[:,:-1])
            return max_steps-cells_visited-sens_dist
        
        elif train_step == 3:
            #wall following
            start_pos = self.data[0]
            pos = np.array((x,y))
            #dist = 0.8 * (np.linalg.norm((start_pos-pos)))
            dist = driven_dist
            max_steps = (600*600)
            sens_dist = 1.0 - np.min(inp_vec[:,:-1])
            return 0.1 * (max_steps/(dist+sens_dist+(0.1*steps)))
    
    
    def get_fitness_finish(self,x,y,quad,finish,steps,inp_vec,driven_dist,cells_visited):
        #calculate the fitness after the finish of the simluation based on the chosen behavior
        
        train_step = utils.TRAIN_STEPS
        
        #calculate the fitness after the simulation finished
        #here we take the final distance to the goal 
        # we substract the number of steps the car spend in the correct half of the field
        # Lastly we give a reward for reaching the goal
        if train_step == 0:
            max_steps = (600*600)
            return (max_steps-cells_visited)
        
        elif train_step == 1: 
            a = np.array((x,y))
            b = np.array((75,75))
            x_q = quad[0]/steps
            y_q = quad[1]/steps
            fitness = 0.1 * (np.linalg.norm(a-b) - (x_q+y_q))
            if finish:
                fitness = 0.01 * fitness
            return fitness
        
        elif train_step == 2:
            max_steps = (600*600)
            sens_dist = np.mean(inp_vec[:,:-1])
            return max_steps-(cells_visited-(0.1*sens_dist))
        
        elif train_step == 3:
            dist = driven_dist
            max_steps = (600*600)
            sens_dist = 1.0 - np.min(inp_vec)
            return 0.1 * (max_steps/(dist+sens_dist+(0.1*steps)))

                            
    def get_fitness_alt(self, x, y):
        # calculates current distance to start of the track from the given position

        if self.data == []:
            print("Error: Data not set.")
            return 0.0

        # get closest point on track
        start_pos_line = self.data[0]

        score = 0.0

        # Iterate over all remaining line points
        for line_point in self.data[1:]:
            lineDir = line_point[0] - start_pos_line[0], line_point[1] - start_pos_line[1]
            line_length = math.sqrt(lineDir[0] ** 2 + lineDir[1] ** 2)
            lineDir = lineDir[0] / line_length, lineDir[1] / line_length

            v = x - start_pos_line[0], y - start_pos_line[1]
            d = v[0] * lineDir[0] + v[1] * lineDir[1]

            # clip d
            if d < 0:
                d = 0
            if d > line_length:
                d = line_length

            # point on line:
            pt = start_pos_line[0] + lineDir[0] * d, start_pos_line[1] + lineDir[1] * d

            dist = math.sqrt((x-pt[0]) ** 2 + (y-pt[1]) ** 2)
            if dist <= self.thickness:
                # we have found the closest line, calculate fitness and return:
                return score + d
            else:
                # add line length to score and keep going
                score += math.sqrt((line_point[0] - start_pos_line[0]) ** 2 + (line_point[1] - start_pos_line[1]) ** 2)
                start_pos_line = line_point[0], line_point[1]
        
        # no point on a line was found, return 0 (car was out of track)
        return 0.0