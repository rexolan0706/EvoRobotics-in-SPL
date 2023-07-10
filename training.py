import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import math

import numpy as np

from evolutional_ai import car
from evolutional_ai import utils
from evolutional_ai import brain_alt as brain
from evolutional_ai import track

if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 12)

    track = track.Track(utils.TRAINING_TRACK)
    track.load()

    start_point, direction = track.get_start_info()

    window = pygame.display.set_mode(utils.SIZE)

    # Init cars
    cars = []
    for i in range(utils.CARS_PER_GEN):
        cars.append(car.Car(start_point, direction, track.image))
    
    goal = False
    finished = False
    restart = False
    iterations = 0
    iterations_after_crash = 0
    generation = 1

    # Overall
    best_fitness = np.inf
    best_fitness_i = None

    # This generation
    curr_best_fitness = np.inf
    curr_best_fitness_i = None

    if utils.INIT_FROM_MODEL:
        for i,c in enumerate(cars):
            c.model.load_from_file(utils.INIT_MODEL)
            if i > 1:
                # mutate all but one
                c.model.set_weights(brain.mutate(c.model.get_weights()))

        if not utils.MAX_ITERATIONS_FROM_TRACK_LENGTH:
            utils.MAX_ITERATIONS += utils.INIT_START_GENERATION * utils.MAX_ITERATIONS_ADD
        generation = utils.INIT_START_GENERATION    


    if utils.MAX_ITERATIONS_FROM_TRACK_LENGTH:
        utils.MAX_ITERATIONS = int(1.5 * track.get_length())
    gen_fitness = []
    fitness_all = []

    while not finished:
        #print(best_fitness)
        pygame.display.set_caption(f"Generation {generation} - Fitness: {best_fitness}")
        window.fill(utils.COL_BLACK)
        #print(iterations)
        if generation >40:
            finished = True

        if restart:
            print(generation)
            generation += 1

            for c in cars:
                gen_fitness.append(c.fitness)
            
            if not utils.MAX_ITERATIONS_FROM_TRACK_LENGTH:
                utils.MAX_ITERATIONS += utils.MAX_ITERATIONS_ADD
            
            iterations = 0
            iterations_after_crash = 0
            # sort vector of cars based on fitness
            cars_sorted = sorted(cars, key=lambda c: c.fitness)

            # save best fitness
            if utils.AUTO_SAVE_BEST_GENOME:
                cars[best_fitness_i].model.save_to_file(utils.AUTO_SAVE_FILENAME)

            # best cars at the front, use first half + mutated for second one
            half = len(cars) // 2

            for i in range(0, half, 2):
                g1, g2 = brain.cross_over(cars_sorted[0].model.get_weights(), cars_sorted[1].model.get_weights())
                cars_sorted[half + i].model.set_weights(g1)
                cars_sorted[half + i + 1].model.set_weights(g2)
            cars_sorted[-1].model.__init__() # create a random car


            track.load()
            start_point, direction = track.get_start_info()
            # reset all cards
            for i,c in enumerate(cars_sorted):
                c.reset(start_point, direction)
                c.model.set_weights(brain.mutate(c.model.get_weights())) # mutate genes
            cars = cars_sorted

            restart = False
            curr_best_fitness = np.inf
            curr_best_fitness_i = None
            fitness_all.append(gen_fitness)
            gen_fitness = []

        # consume all pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True
            elif event.type == pygame.KEYDOWN:
                # If pressed key is ESC quit program
                if event.key == pygame.K_ESCAPE:
                    finished = True
                elif event.key == ord('s'):
                    if not curr_best_fitness_i == None:
                        cars[curr_best_fitness_i].model.save_to_file(utils.AUTO_SAVE_FILENAME)

        # add background to window
        window.blit(track.image, (0,0))

        # Update all cars
        crashed = 0
        for i,c in enumerate(cars):
            if c.crashed or c.finished:
                # skip crashed cars
                crashed += 1
                if crashed == len(cars):
                    restart = True
                    break
            
            if c.finished:
                goal = True
            
            # calculate fitness
            c.quads = track.calc_quadrant(c.position[0], c.position[1],c.quads)
            c.fitness = track.get_fitness(c.position[0], c.position[1],c.quads,c.steps,c.compute_input_vector(),c.driven_dist,c.visited_cells) 
            c.steps +=1
            #- utils.TIME_PENALTY * iterations
            if c.fitness < best_fitness:
                best_fitness = c.fitness
                best_fitness_i = i

            if c.fitness < curr_best_fitness:
                curr_best_fitness = c.fitness
                if curr_best_fitness_i == None:
                    pass
                else:
                    # set this color back to red
                    cars[curr_best_fitness_i].setRed()
                curr_best_fitness_i = i
                cars[i].setGreen()

            c.perform_autonomous_action()
            c.update()
            if not i == curr_best_fitness_i:
                c.draw(window)
        if not curr_best_fitness_i == None:
            cars[curr_best_fitness_i].draw(window, True) # draw best car on top and with lines

            # Get inputs for this car as well
            # then turn on the wall following feature
            inp = cars[curr_best_fitness_i].compute_input_vector()
            inp_text = "Inputs: " + "{:.3f} " * len(utils.CAR_INPUT_ANGLES.split(","))
            text1 = font.render(inp_text.format(*inp[0]), True, utils.COL_WHITE)
            window.blit(text1, (500,15))
            outputsList = inp.tolist()
            leftDis = outputsList[0][0]
            leftDis1 = outputsList[0][1]
            leftDis2 = outputsList[0][2]
            # print(outputsList)
            if crashed == len(cars):
                #if leftDis + leftDis1 + leftDis2 > 0.21:
                print('nazdike', iterations)
                restart = True
                iterations = 0
                goal = False
            text2 = font.render("Velocity: {:.3f}, Angle Velocity: {:.3f}".format(cars[curr_best_fitness_i].velocity, cars[curr_best_fitness_i].angle_velocity), True, utils.COL_WHITE)
            window.blit(text2, (500, 30))

        if not utils.TRAIN_STEPS ==2:
            if crashed >= len(cars)/2:
                iterations_after_crash += 1
        print(iterations_after_crash,iterations)
        iterations += 1

        for c in cars:
            c.fitness = track.get_fitness_finish(c.position[0],c.position[1],c.quads,c.finished,c.steps,c.compute_input_vector(),c.driven_dist,c.visited_cells) 
            #+ utils.TIME_PENALTY * iterations

        # Force restart after MAX_ITERATIONS iterations
        if iterations >= utils.MAX_ITERATIONS or iterations_after_crash >= utils.MAX_ITERATIONS_AFTER_CRASH:
            restart = True
        pygame.display.flip()

    np.save("./metrics/config_142.npy",np.array(fitness_all))
    np.save("./models/config_142_head.npy",cars[best_fitness_i].model.get_weights()[0])
    np.save("./models/config_142_class.npy",cars[best_fitness_i].model.get_weights()[1])

    # Quit
    pygame.quit()
