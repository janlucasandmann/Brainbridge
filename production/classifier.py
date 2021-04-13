""" Classify incoming EEG signals in real-time """

import numpy as np
import scipy.signal
import math
import time
from bitalino import BITalino
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

import helpers as hp

# How long should one interval be?
interval_time = 2

# How many intervals should be measured?
measurement_intervals = 50

# Define list of events to initially train the classififer with real labels
events_initial = [0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0]
#events_initial = [0,0,0,1,1,1]

# Measurements are too noisy in the beginning --> wait a few seconds
waiting_intervals = 10

# How many intervals for initialization? (including waiting intervals)
initialization_intervals = len(events_initial) + waiting_intervals

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
X_indices = []








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
    global X_indices
    
    print("step 1 / 7")
    data_raw = generateInputData(data_raw_one, data_raw_two) # From now on all sensors in one list
    data_split = splitData(data_raw)
    #print(len(data_split))
    #print(len(data_split[0]), "data split...")
    #print(len(data_split[0][0]), "data split......")
    print("step 2 / 7")
    data, events = hp.removeArtifacts(data_split, events_raw, 10000, -10, 370, -1)
    print("step 3 / 7")
    # Get Extreme Points
    mini, maxi = hp.extremePointsCorrelation(data, events, 10)
    print(len(data))
   # print(len(data[0]))
    print("step 4 / 7")
    # Get frequency bands
    cores_real_numbers = hp.getFrequencies(1,3, data)
    print("step 5 / 7")
    # Combine features
    X_whole_input = cores_real_numbers + mini + maxi
    print("step 6 / 7")
    # Feature reduction
    X_recuded, X_indices = hp.featureReduction(X_whole_input, 0, events)
    
    X_train, target = hp.generateTrainingSet(X_recuded, events)
    print("Events", events, target)
    print("step 7 / 7")
    clf.fit(X_train, target)
    #print("Accuracy:",metrics.accuracy_score(X_train, target))
    

def main(data_raw_one, data_raw_two):
    global X_indices
    
    data_raw = generateInputData(data_raw_one, data_raw_two)
    data_split = data_raw # Data already split...
    
    extremeValue = False
    
    if 1018 in data_split[0]:
        extremeValue = True
        
    if extremeValue:
        print("warning")
        
    else:
        data = data_split
        
        # Get Extreme Points
        mini, maxi = hp.extremePointsCorrelationMain(data, 10)
        
        # Get frequency bands
        cores_real_numbers = hp.getFrequencies(1,3, data)

        # Combine features
        X_whole_input = cores_real_numbers + mini + maxi

        X_reduced = hp.flatten_list(reduceFeatures(X_whole_input, X_indices))
        
      #  X_predict = pd.DataFrame(hp.flatten_list(X_reduced))
        
        prediction = clf.predict([X_reduced])
        
        if prediction == 1:
            print("SIGNAL!!!!!!! ", prediction)
        else:
            print(prediction)
            
            
init_count = 0
print("Entering initialization phase")
model_train_counter = 0

while (end - start) <= running_time:
   
    raw = device.read(nSamples).tolist()
    input_data_one = []
    input_data_two = []
    
    
    
    for i in range(0, nSamples):
        input_data_one.append(raw[i][5])
        input_data_two.append(raw[i][6])
    
    if init_count < initialization_intervals: # If we are in initialization phase
        if 1018 in input_data_one:
            print("warning")
        
        
        
        if init_count > waiting_intervals:
            c = 0
            for i in input_data_one:
                xs_one_initialization.append(i)
                xs_two_initialization.append(input_data_two[c])
                c += 1
                
        if len(xs_one_initialization) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            if init_count > waiting_intervals:
                if events_initial[init_count - waiting_intervals] == 1:
                    print("HOCH")
                else:
                    print("runter")
            init_count += 1

            
        
    else: # If we are in production phase
        if model_train_counter == 0:
            print("jojo")
            calibrateModel(xs_one_initialization, xs_two_initialization, events_initial)
            model_train_counter += 1
        c = 0
        for i in input_data_one:
            xs_one_row.append(i)
            xs_two_row.append(input_data_two[c])
            
        if len(xs_one_row) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            xs_one.append(xs_one_row)
            xs_two.append(xs_two_row)
            main([xs_one_row], [xs_two_row])
            xs_one_row = []
            xs_two_row = []
            
            
hp.write([events_initial, xs_initialization, xs, pred], ["events_initial", "initialization_data", "production_data", "predictions"], "subject_1")
