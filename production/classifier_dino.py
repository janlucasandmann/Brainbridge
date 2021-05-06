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

# How long should one interval be?
interval_time = 2

# How many intervals should be measured?
measurement_intervals = 100

# Define list of events to initially train the classififer with real labels

#events_initial = [0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0]
#events_initial = [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0]
#events_initial = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0]
#events_initial = [0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0]

events_initial = [1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0]


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

clf=RandomForestClassifier(n_estimators=10)
reg = LinearRegression()
X_indices = []
X_indices_real = []
mini_indices = []
maxi_indices = []
data_saved = []

GLOBAL_CORR_LIMIT = 0.2
GLOBAL_CORR_LIMIT_NUMBER = 50



jump = 0
MAX_WIDTH = 800
MAX_HEIGHT = 400





def generateInputData(data_raw_one, data_raw_two):
    c = 0
    data_raw = []
    
    for i in data_raw_one:
        data_raw.append([i, data_raw_two[c]])
        c += 1
        
    return data_raw

def splitData(data_raw):
    u = 0
    data_row = []
    data_split = []
    
    for i in data_raw:
        u += 1
        data_row.append(i)
        
        if u == 200:
            data_split.append(data_row)
            u = 0
            data_row = []
            
    return data_split
    
def reduceFeaturesBackup(input_data, X_indices):
    res = []
    
    c = 0
    for i in input_data:
        if c in X_indices:
            res.append(i)
        c += 1
            
    return res

def reduceFeatures(input_data, X_indices):
    res = []
    
    c = 0
    for i in input_data:
        if c in X_indices:
            res.append(i)
        c += 1
            
    return res


def calibrateModel(data_raw_one, data_raw_two, events_raw):
    global clf
    global reg
    global X_indices
    global mini_indices
    global maxi_indices
    global data_saved
    global X_indices_real
    global GLOBAL_CORR_LIMIT
    global GLOBAL_CORR_LIMIT_NUMBER
    
    
    print("step 0 / 7")
    pygame.init()
    pygame.display.set_caption('Brainbridge Dino')
    
    print("step 1 / 7")
    data_raw = generateInputData(data_raw_one, data_raw_two) # From now on all sensors in one list 
    data_split = splitData(data_raw) # [ [ [x,y], [x,y], ... ], ... ]
    #print(len(data_split))
    #print(len(data_split[0]), "data split...")
    #print(len(data_split[0][0]), "data split......")
    print("step 2 / 7")
    data, events = hp.removeArtifacts(data_split, events_raw, 10000, -10, 370, -1)
    print("step 3 / 7")
    # Get Extreme Points
    mini, maxi, mini_indices, maxi_indices = hp.extremePointsCorrelation(data, events, 10)
    
    
    print("MINI: ", mini, len(mini))
    print("MAXI: ", maxi, len(maxi))
    
    print("step 4 / 7")
    # Get frequency bands
    cores_real_numbers = hp.getFrequencies(1,49, data)
    print("step 5 / 7")
    # Combine features
    X_whole_input = cores_real_numbers #+ mini + maxi # ??????????????????? DAS GEHT SO NICHT ........
    print("step 6 / 7")
    # Feature reduction
    X_reduced, X_indices = hp.featureReduction(X_whole_input, 0.12, events)
    #print("TESTETSTETETST 01010101: ", cores_real_numbers)
    
    X_reduced_res = []
    
    c = 0
    for i in X_reduced:
        X_reduced_res_row = i
        for x in np.transpose(mini)[c]:
            X_reduced_res_row.append(x)
        for x in np.transpose(maxi)[c]:
            X_reduced_res_row.append(x)
        X_reduced_res.append(X_reduced_res_row)
        c += 1
      
  
    corr_sort_array = []
    
    c = 0
    for i in np.transpose(X_reduced_res):
        corr_sort_array.append([c, math.sqrt(np.corrcoef(i, events)[0][1] ** 2)])
        c += 1
    
    corr_sorted_array = sorted(corr_sort_array,key=lambda x: x[1])
    X_indices_real_input = corr_sorted_array[::-1]
    
    X_indices_real = np.transpose(X_indices_real_input)[0][0:GLOBAL_CORR_LIMIT_NUMBER]
        
    X_reduced_res_real = []
    
    for epoch in X_reduced_res:
        X_reduced_res_real_row = []
        c = 0
        for x in epoch:
            if c in X_indices_real:
                X_reduced_res_real_row.append(x)
            c += 1
        X_reduced_res_real.append(X_reduced_res_real_row)
            
    
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
    
    #print("Data raw one: ", data_raw_one)
    #print("Data raw two: ", data_raw_two)
    
    global X_indices
    global mini_indices
    global maxi_indices
    
    global X_indices_real
    
    data_raw = generateInputData(data_raw_one, data_raw_two)
    data_split = [data_raw] # Data already split...
    
    extremeValue = False
    
    if 1018 in np.transpose(data_split)[0]:
        extremeValue = True
        
    if extremeValue:
        print("warning")
        
    else:
        data = data_split
        print("Data: ", data, len(data))
        
        # Get Extreme Points
        mini, maxi = hp.extremePointsCorrelationMain(data, 10, mini_indices, maxi_indices)
        
        #print("Mini zuzu, Maxi zuzuzu: ", mini, maxi, mini_indices, maxi_indices)
        
        # Get frequency bands
        cores_real_numbers = hp.getFrequencies(1,49, data)
        #print("COOOOOOOOORRRRREESSS 01010101", cores_real_numbers)
        #print("cores_real_numbers: ", cores_real_numbers)

        # Combine features
        X_whole_input = cores_real_numbers
        
        #print("X_whole_input: ", X_whole_input[0:20], len(X_whole_input))

        X_reduced = reduceFeatures(X_whole_input, X_indices)
        #X_reduced = X_reduced + mini + maxi
        #print("X_reduced: ", X_reduced)
        #print("Len X_reduced: ", len(X_reduced))
        #print("X_reduced: ", X_reduced[len(X_reduced) - 1])
        #print("Len X_reduced: ", X_reduced[len(X_reduced) - 1])
        
        X_predict = hp.flatten_list(X_reduced)
        
        for i in mini:
            for x in i:
                X_predict.append(x)
        for i in maxi:
            for x in i:
                X_predict.append(x)
                
                
        X_reduced_res_real = []
    
        c = 0
        for x in X_predict:
            if c in X_indices_real:
                X_reduced_res_real.append(x)
            c += 1
        
        print("X PREDICT: ", X_reduced_res_real, len(X_reduced_res_real))
        #    print("UIUO", data)
        #   print("UIUO", mini, maxi)
        #   print("UIUO", cores_real_numbers)
        # print("UIUO", X_whole_input)
        #    print("--------------------")
        #   print("X WHOLE INPUT: ", len(X_whole_input))
        #  print("X INDICES: ", X_indices)
        ##print("REDUCE FEATURES: ", reduceFeatures(X_whole_input, X_indices))
        #print(len(X_reduced))
        
        
        
        prediction = clf.predict([X_reduced_res_real])
        pred_linreg = reg.predict([X_reduced_res_real])
        
        pred.append([prediction, pred_linreg])
        
        
        #if prediction == 1:
            #print("SIGNAL!!!!!!! ", prediction)
        #else:
            #print("No Signal. ", prediction)
            
        print("RANDOM FOREST PREDICTION: ", prediction)
        print("LIN REG PREDICTION: ", pred_linreg)
        
        if prediction == 1:
            jump = 1
            
            
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
        if 1018 in input_data_one:
            print("warning")
        
        c = 0
        for i in input_data_one:
            xs_one_initialization.append(i)
            xs_two_initialization.append(input_data_two[c])
            c += 1
                
        if len(xs_one_initialization) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            init_count += 1
            if init_count < len(events_initial):
                if events_initial[init_count] == 1:
                    print("HOCH")
                else:
                    print("runter")
            

            
        
    else: # If we are in production phase
        if model_train_counter == 0:
            print("jojo")
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
            dino_y -= 10.0
        elif not is_go_up and not is_bottom:
            dino_y += 10.0

        # dino top and bottom check
        if is_go_up and dino_y <= jump_top:
            is_go_up = False

        if not is_bottom and dino_y >= dino_bottom:
            is_bottom = True
            dino_y = dino_bottom

        # tree move
        tree_x -= 12.0
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
            
        if len(xs_one_row) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            xs_one.append(xs_one_row)
            xs_two.append(xs_two_row)
            main(xs_one_row, xs_two_row)
            xs_one_row = []
            xs_two_row = []
            
            
hp.write([events_initial, xs_one_initialization, xs_two_initialization, xs_one, xs_two, pred], ["events_initial", "initialization_data_one", "initialization_data_two", "production_data_one", "production_data_two", "predictions"], "subject_1")
