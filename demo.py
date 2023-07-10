import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import time

from evolutional_ai import car
from evolutional_ai import utils
from evolutional_ai import track

import numpy as np


if __name__ == "__main__":
    pygame.init()

    font = pygame.font.Font(pygame.font.get_default_font(), 12)

    track = track.Track(utils.DEMO_TRACK)
    track.load() # load the track from disk

    start_point, direction = track.get_start_info()

    window = pygame.display.set_mode(utils.SIZE)

    cars = [car.Car(start_point, direction, track.image) for x in range(4)]

    if utils.DEMO_BRAIN == "":
        print("Error: Please set the file to load inside the configuration.")
    else:
        for c in cars:
            c.model.load_from_file(utils.DEMO_BRAIN)

    finished_all = []
    crashed_all = []
    for pos in [[200,200],[500,500],[500,200],[200,500],[365,365]]:
        quit = False
        restart = False
        crashed = 0
        finished = 0
        angle = 0
        for c in cars:
            dir = [90,180,270,360]
            c.reset(pos,dir[angle])
            angle+=1
        for x in range(1000):
            window.fill(utils.COL_BLACK)
            if restart:
                break
                
            
            # Consume all events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
                elif event.type == pygame.KEYDOWN:
                    # If pressed key is ESC quit program
                    if event.key == pygame.K_ESCAPE:
                        quit = True
            window.blit(track.image, (0,0))

            for c in cars:
                c.perform_autonomous_action()
                c.update()

                c.draw(window, True)
                

            text = font.render("Velocity: {:.3f}, Angle Velocity: {:.3f}".format(cars[0].velocity, cars[0].angle_velocity), True, utils.COL_WHITE)
            window.blit(text, (480, 15))

            pygame.display.flip()
        for c in cars:
                    if c.crashed:
                        crashed += 1
                    if c.finished:
                        finished +=1
        crashed_all.append(crashed)
        finished_all.append(finished)
    print(crashed_all,finished_all)
    pygame.quit()