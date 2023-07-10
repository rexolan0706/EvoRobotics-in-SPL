import numpy as np
import os
import pygame
import configparser
import math

config = configparser.ConfigParser()
config.read('config.ini') # next to calling file

# Read all configurations
# General
AUTO_SAVE_BEST_GENOME = config.getboolean("General", "AUTO_SAVE_BEST_GENOME_EVERY_GEN", fallback=True)
AUTO_SAVE_FILENAME = config.get("General", "AUTO_SAVE_FILEPATH", fallback="autosave.txt")

# Training
TRAINING_TRACK = config.get("Training", "TRACK", fallback="training")
CARS_PER_GEN = config.getint("Training", "CARS_PER_GEN", fallback=32)
MAX_ITERATIONS = config.getint("Training", "MAX_ITERATIONS", fallback=100)
MAX_ITERATIONS_AFTER_CRASH = config.getint("Training", "MAX_ITERATIONS_AFTER_CRASH", fallback=100)
TIME_PENALTY = config.getfloat("Training", "TIME_PENALTY", fallback=0.01)
MAX_ITERATIONS_FROM_TRACK_LENGTH = config.getboolean("Training", "MAX_ITERATIONS_FROM_TRACK_LENGTH", fallback=True)
MAX_ITERATIONS_ADD = config.getint("Training", "MAX_ITERATIONS_ADD", fallback=10)
INIT_FROM_MODEL = config.getboolean("Training", "INIT_FROM_MODEL", fallback=True)
INIT_MODEL = config.get("Training", "INIT_MODEL", fallback="autosave.txt")
INIT_START_GENERATION = config.getint("Training", "INIT_START_GENERATION", fallback=1)
TRAIN_STEPS = config.getint("Training","TRAIN_STEPS",fallback=1)

# Display
SIZE = config.getint("Display", "SIZE_X", fallback=800), config.getint("Display", "SIZE_Y", fallback=600)

# Car
CAR_WIDTH = config.getint("Car", "CAR_WIDTH", fallback=6)
CAR_LENGTH = config.getint("Car", "CAR_HEIGHT", fallback=12)
MAX_DIST = config.getint("Car", "MAX_DIST", fallback=100)
MAX_TURN_VELOCITY = config.getfloat("Car", "MAX_TURN_VELOCITY", fallback=5.0)
CAR_INPUT_ANGLES = config.get("Car", "INPUT_ANGLES", fallback="-90,-45,0,45,90")

# Brain
CROSS_OVER_PROB = config.getfloat("Brain", "CROSS_OVER_PROB", fallback=0.2)
MUTATION_PROB = config.getfloat("Brain", "MUTATION_PROB", fallback=0.5)
HIDDEN_LAYERS = config.get("Brain", "HIDDEN_LAYERS", fallback="4,3")

# Demo
DEMO_BRAIN = config.get("Demo", "BRAIN_TO_LOAD", fallback="")
DEMO_TRACK = config.get("Demo", "TRACK", fallback="training")

# Draw
DRAW_TRACK = config.get("Draw", "TRACK", fallback="training")

# Track
THICKNESS = config.getint("Track", "THICKNESS", fallback=2)

# Color definitions
COL_BLACK = pygame.color.Color(0,0,0)
COL_RED = pygame.color.Color(255,0,0)
COL_GREEN = pygame.color.Color(0,255,0)
COL_BLUE = pygame.color.Color(0,0,255)
COL_WHITE = pygame.color.Color(255,255,255)

# Paths for folders
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # twice to climb up
TRACKS_PATH = os.path.join(MAIN_PATH, "tracks")
MODELS_PATH = os.path.join(MAIN_PATH, "models")

#Light
LIGHT_X = config.getint("Light","LIGHT_X",fallback=300)
LIGHT_Y = config.getint("Light","LIGHT_Y",fallback=300)