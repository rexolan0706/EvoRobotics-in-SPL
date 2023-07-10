# Evolutionary Robotics in Software Product Lines

## Installation
Clone this repository and install the requirements.txt
## Usage
### Trianing
To train the behaviors change the TRAIN_STEPS Parameter in the Config.ini file. 
Also set the corresponding arenas in the TRACK Parameter under the point "Training". The arenas can be found in the track folder.
|Behavior|Training step|Arena|
---|---|---|
Exploration|0|exploraion_train|
Homing|1|homing_train|
Wall follwing|2|circle|
Obstacle avoidance|3|avoidance_train|

If a new model should be trained, then set INIT_FROM_MODEL to False.
If a previous model should be trained then set INIT_FROM_MODEL to True, while also setting INIT_MODEL to the model path.

After setting all parameters just execute train.py

### Testing
For testing the model path BRAIN_TO_LOAD as well as the track TRACK under the DEMO point have to be set.
Then it can be tested by executing demo.py.



This project is licensed under the terms of the MIT license.
