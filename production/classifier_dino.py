""" Classify incoming EEG signals in real-time """

import numpy as np
import scipy.signal
import math
import time
from bitalino import BITalino
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
import helpers as hp
import pygame
import sys
import genetic as gen

# Settings for Dino game
jump = 0
MAX_WIDTH = 800
MAX_HEIGHT = 400

# How long should one interval be?
interval_time = 2

# How often should the incoming data be classified within the invterval time?
GLOBAL_CLASSIFICATION_FREQUENCY = 5

# How many intervals should be measured?
measurement_intervals = 100

# Define list of events to initially train the classififer with real labels

#events_initial = [0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0]
#events_initial = [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0]
#events_initial = [0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0]
#events_initial = [0,0,1,1]
#events_initial = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0]
events_initial = [1,0,0,1,0]
#events_initial = [1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0]


# How many intervals for initialization?
initialization_intervals = len(events_initial)

# Get total running time of program in seconds
running_time = (len(events_initial) + measurement_intervals) * interval_time

# Store initialization data
xs_one_initialization = []
xs_one_initialization_row = []
xs_two_initialization = []
xs_two_initialization_row = []

# Store production data
xs_one = []
xs_one_row = []
xs_two = []
xs_two_row = []

# Store predictions
pred = []

# Connect to BITalino device
macAddress = "/dev/tty.BITalino-51-FB-DevB"

# Define preferences for BITalino
batteryThreshold = 30
acqChannels = [0, 1]
samplingRate = 100
nSamples = 10
digitalOutput = [1,1]

device = BITalino(macAddress)
device.battery(batteryThreshold)

# Start device
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()

clf=RandomForestClassifier(n_estimators=500)
reg = LinearRegression()
X_indices = []
X_indices_real = []
data_saved = []

GLOBAL_CORR_LIMIT = 0.2
GLOBAL_CORR_LIMIT_NUMBER = len(events_initial) + 2

GLOBAL_JUMP_MEMORY_ONE = 0
GLOBAL_JUMP_MEMORY_TWO = 0

GLOBAL_UPPER_WARNING_LIMIT_ONE = 10180
GLOBAL_UPPER_WARNING_LIMIT_TWO = 380

GLOBAL_PREDICTION_NUMBER = 1
GLOBAL_EVALUATE_PREDICTION_INTERVALS = 5
GLOBAL_EVALUATE_PREDICTION_THRESHOLD = 3
GLOBAL_EVALUATE_PREDICTION_BLOCKER = 0



def calibrateModel(data_raw_one, data_raw_two, events_raw):
    
    # Load global variables
    global clf
    global reg
    global X_indices
    global data_saved
    global X_indices_real
    global GLOBAL_CORR_LIMIT
    global GLOBAL_CORR_LIMIT_NUMBER
    global GLOBAL_UPPER_WARNING_LIMIT_ONE
    global GLOBAL_UPPER_WARNING_LIMIT_TWO
    
    # Initialize dino game
    pygame.init()
    pygame.display.set_caption('Brainbridge Dino')
    
    data_raw = hp.generateInputData(data_raw_one, data_raw_two) # From now on all sensors in one list 
    data_split = hp.splitData(data_raw) # [ [ [x,y], [x,y], ... ], ... ]

    data, events = hp.removeArtifacts(data_split, events_raw, GLOBAL_UPPER_WARNING_LIMIT_ONE, GLOBAL_UPPER_WARNING_LIMIT_TWO)
    
    # Get Extreme Points
    mini, maxi = hp.extremePointsCorrelation(data, events, 10)
    
    # Get frequency bands
    cores_real_numbers = hp.getFrequenciesPredefined(data)
    
    X_reduced_res = hp.concatenateFeatures(cores_real_numbers, mini, maxi)
    
    #print("len X_reduced_res: XXXXXdfgfdgdfgdfdfgwesfhdsjksdhfdsjkXXX ", len(X_reduced_res), len(X_reduced_res[0]), len(mini[0][0]), len(maxi[0][0]), len(cores_real_numbers[0][0]))
    
    #X_reduced_res_real, X_indices_real = hp.getFeaturesBasedOnKBest(X_reduced_res, events, GLOBAL_CORR_LIMIT_NUMBER)
    
    # ------------------------------------------------------------------------------------------------------------------------------------- #
    
    print("Starting genetic algorithm")
    
    X_indices_real = gen.simulateEvolution(np.transpose(X_reduced_res), 8, 24, 2, 2, 0.5, (len(events) - 2), 100, events)
    
    print("Ended genetic algorithm")
    print(X_indices_real)
    
    X_reduced_res_real = []
    
    for row in X_reduced_res:
        res_row = []
        c = 0
        for i in row:
            if c in X_indices_real:
                res_row.append(i)
            c += 1
        X_reduced_res_real.append(res_row)
    
    # ------------------------------------------------------------------------------------------------------------------------------------- #
    
    X_train, target = hp.generateTrainingSet(X_reduced_res_real, events)
    print("X Train: ", X_train)
    print("Events", events, target)
    print("X indices: ", X_indices)
    print("step 7 / 7")
    clf.fit(X_train, target)
    reg = LinearRegression().fit(X_train, target)
    print("Accuracy:",metrics.accuracy_score(clf.predict(X_train), target))
    print("Predictions: ", clf.predict(X_train))

def main(data_raw_one, data_raw_two):
    global jump    
    global X_indices 
    global X_indices_real
    global GLOBAL_JUMP_MEMORY_ONE
    global GLOBAL_JUMP_MEMORY_TWO
    global GLOBAL_PREDICTION_NUMBER
    global GLOBAL_EVALUATE_PREDICTION_INTERVALS
    global GLOBAL_EVALUATE_PREDICTION_THRESHOLD
    
    data_raw = hp.generateInputData(data_raw_one, data_raw_two)
    data_split = [data_raw] # Data already split...
    
    extremeValue = False
    
    if GLOBAL_UPPER_WARNING_LIMIT_ONE in np.transpose(data_split)[0]:
        extremeValue = True
        
    if extremeValue:
        print("warning")
        
    else:
        data = data_split
        print("Data: ", data, len(data))
        print("X indices kumbaya: ", X_indices_real)
        
        # Get Extreme Points
        mini, maxi = hp.extremePointsCorrelationMain(data, 10)
        # Get frequency bands
        cores_real_numbers = hp.getFrequenciesPredefined(data)
        # Concatenate both lists to one prediction list
        X_predict = hp.concatenateFeaturesMain(cores_real_numbers, mini, maxi, X_indices_real)
        
        print("mini: ", mini)
        print("maxi: ", maxi)
        print("X_predict: ", X_predict)
        #print("len x predict tzrutzuertzeurzt nicht nuct: ", len(X_predict), len(cores_real_numbers), len(mini[0]), len(maxi[0]), len(cores_real_numbers))
        
        # Get predictions
        prediction = clf.predict([X_predict])
        #pred_linreg = reg.predict([X_predict])
        
        pred.append(prediction)
            
        print(GLOBAL_PREDICTION_NUMBER, " RANDOM FOREST PREDICTION: ", prediction)
        #print(GLOBAL_PREDICTION_NUMBER, "LIN REG PREDICTION: ", pred_linreg)
        GLOBAL_PREDICTION_NUMBER += 1
        
        
        #TODO: Diese geringeren Intervalle neu durchdenken ...
        #if hp.evaluatePrediction(pred, GLOBAL_EVALUATE_PREDICTION_INTERVALS, GLOBAL_EVALUATE_PREDICTION_THRESHOLD):
        #    if GLOBAL_EVALUATE_PREDICTION_BLOCKER == 0:
        #        jump = 1
                
        #GLOBAL_EVALUATE_PREDICTION_BLOCKER += 1
            
        #if GLOBAL_EVALUATE_PREDICTION_BLOCKER == 5:
        #    GLOBAL_EVALUATE_PREDICTION_BLOCKER = 0
        
        #if prediction == 1 and GLOBAL_JUMP_MEMORY_ONE == 0 and GLOBAL_JUMP_MEMORY_TWO == 0:
        #    jump = 1
        #    GLOBAL_JUMP_MEMORY_TWO = GLOBAL_JUMP_MEMORY_ONE
        #    GLOBAL_JUMP_MEMORY_ONE = 1
        #else:
        #    GLOBAL_JUMP_MEMORY_TWO = GLOBAL_JUMP_MEMORY_ONE
        #    GLOBAL_JUMP_MEMORY_ONE = 0
            
        
            
            
init_count = 0
print("Entering initialization phase")
model_train_counter = 0

# DINO STUFF ...
# set screen, fps
screen = pygame.display.set_mode((MAX_WIDTH, MAX_HEIGHT))
fps = pygame.time.Clock()

# dino
imgDino1 = pygame.image.load('./dino1.png')
imgDino2 = pygame.image.load('./dino2.png')
dino_height = imgDino1.get_size()[1]
dino_bottom = MAX_HEIGHT - dino_height
dino_x = 50
dino_y = dino_bottom
jump_top = 200
leg_swap = True
is_bottom = True
is_go_up = False

# tree
imgTree = pygame.image.load('tree.png')
tree_height = imgTree.get_size()[1]
tree_x = MAX_WIDTH
tree_y = MAX_HEIGHT - tree_height






while (end - start) <= running_time:
    end = time.time()
   
    raw = device.read(nSamples).tolist()
    input_data_one = [] 
    input_data_two = []
    
    
    
    for i in range(0, nSamples):
        input_data_one.append(raw[i][5])
        input_data_two.append(raw[i][6])
    
    if init_count < initialization_intervals: # If we are in initialization phase
        if GLOBAL_UPPER_WARNING_LIMIT_ONE in input_data_one:
            print("warning")
        
        c = 0
        for i in input_data_one:
            xs_one_initialization.append(i)
            xs_two_initialization.append(input_data_two[c])
            c += 1
                
        if len(xs_one_initialization) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            init_count += 1
            print("TERROR 1", xs_one_initialization)
            print("TERROR 2", xs_one_initialization)
            
            if init_count < len(events_initial):
                if events_initial[init_count] == 1:
                    print("HOCH")
                else:
                    print("runter")
            

            
        
    else: # If we are in production phase
        if model_train_counter == 0:
            calibrateModel(xs_one_initialization, xs_two_initialization, events_initial)
            model_train_counter += 1
            
            
            
        screen.fill((255, 255, 255))

        if jump == 1:
            if is_bottom:
                is_go_up = True
                is_bottom = False
            jump = 0

        # dino move
        if is_go_up:
            dino_y -= 15.0
        elif not is_go_up and not is_bottom:
            dino_y += 15.0

        # dino top and bottom check
        if is_go_up and dino_y <= jump_top:
            is_go_up = False

        if not is_bottom and dino_y >= dino_bottom:
            is_bottom = True
            dino_y = dino_bottom

        # tree move
        tree_x -= 8.0
        if tree_x <= 0:
            tree_x = MAX_WIDTH

        # draw tree
        screen.blit(imgTree, (tree_x, tree_y))

        # draw dino
        if leg_swap:
            screen.blit(imgDino1, (dino_x, dino_y))
            leg_swap = False
        else:
            screen.blit(imgDino2, (dino_x, dino_y))
            leg_swap = True

        # update
        pygame.display.update()
        #fps.tick(30)
            
        c = 0
        for i in input_data_one:
            xs_one_row.append(i)
            xs_two_row.append(input_data_two[c])
            c += 1
            
        if len(xs_one_row) % 200 == 0: # After n seconds of measurement
            xs_one.append(xs_one_row)
            xs_two.append(xs_two_row)
            
            # TODO: Abstände der Messungen verringern!!!!!
            
            print("Kumbayayayayaa", len(xs_one_row), len(xs_two_row))
            
            #main(xs_one_row[(-interval_time * samplingRate):], xs_two_row[(-interval_time * samplingRate):])
            main(xs_one_row, xs_two_row)
            
            xs_one_row = []
            xs_two_row = []
            
            
hp.write([events_initial, xs_one_initialization, xs_two_initialization, xs_one, xs_two, pred], ["events_initial", "initialization_data_one", "initialization_data_two", "production_data_one", "production_data_two", "predictions"], "subject_1")
