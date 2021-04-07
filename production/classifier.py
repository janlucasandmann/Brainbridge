""" Classify incoming EEG signals in real-time """

import numpy as np
import scipy.signal
import math
import time
from bitalino import BITalino
import csv

import helpers as hp

# How long should one interval be?
interval_time = 2

# How many intervals should be measured?
measurement_intervals = 50

# Define list of events to initially train the classififer with real labels
events_initial = [0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0]

# Measurements are too noisy in the beginning --> wait a few seconds
waiting_intervals = 10

# How many intervals for initialization? (including waiting intervals)
initialization_intervals = len(events_initial) + waiting_intervals

# Get total running time of program in seconds
running_time = len(events_initial + measurement_intervals) * interval_time

# Store initialization data
xs_initialization
xs_initialization_row

# Store production data
xs = []
xs_row = []

# Store predictions
pred = []

# Connect to BITalino device
macAddress = "/dev/tty.BITalino-51-FB-DevB"

# Define preferences for BITalino
batteryThreshold = 30
acqChannels = [0]
samplingRate = 100
nSamples = 10
digitalOutput = [1,1]

device = BITalino(macAddress)
device.battery(batteryThreshold)

# Start device
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()

while (end - start) <= running_time:
    raw = device.read(nSamples).tolist()
    input_data = []
    
    init_count = 0
    
    for i in range(0 ..< nSamples):
        input_data.append(raw[i][5])
    
    if init_count < initialization_intervals: # If we are in initialization phase
        
        for i in input_data:
            xs_initialization_row.append(i)
            
        if len(xs_initialization_row) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            init_count += 1
            if init_count >= waiting_intervals:
                xs_initialization.append(xs_initialization_row)
                main(xs_initialization)
                
            xs_initilization_row = []
        
    else: # If we are in production phase
        
        for i in input_data:
            xs_row.append(i)
            
        if len(xs_row) % (interval_time * samplingRate) == 0: # After n seconds of measurement
            xs.append(xs_row)
            main(xs_row)
            
            
hp.write(xs_initialization, xs, pred)
                
            
            
        