# lstm autoencoder recreate sequence
import os
from numpy import array
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
import numpy as np
import scipy.signal
import math
import time
from bitalino import BITalino
import csv
import matplotlib as plt

# Training Array for initialization phase
events = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]

# Load model --> Pretrained in Cloud based on other patients
model = tf.keras.models.load_model('saved_model/prod_model_vRef')
# EEG sensor needs time to make accurate measurements --> wait a certain amount of intervals until measurements are correct
waiting_intervals = 30
# Length of initialization phase
initalization_intervals = len(events) + waiting_intervals
# Running time in seconds
running_time = (len(events) + 90) * 2

# Store initialization data
initializationData = []
initializationDataRow = []

# Store production data
xs = []
xsRow = []

# Store minmax transformed data
alpha_means_min = 0
alpha_means_max = 0

alpha_stds_min = 0
alpha_stds_max = 0

beta_means_min = 0
beta_means_max = 0

beta_stds_min = 0
beta_stds_max = 0

# Store predictions
prodRes = []
predictions = []


# TODO: TOP VAL pro Intervall identifizieren: Korrelation: Geschätzt -10%

# TODO: Messungen erst nach 75-100 Signalen anfangen, da Reaktionszeit...

# TODO: Include Baseline Difference (INtervall: 100 - 160)

# TODO: Baseline Difference mit Average der drei größten und kleinsten Wellenausschläge, statt 1

# TODO: Intervalle in Prod auf 1s reduzieren !!!!!

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "/dev/tty.BITalino-51-FB-DevB"
    
batteryThreshold = 30
acqChannels = [0]
samplingRate = 100
nSamples = 10
digitalOutput = [1,1]

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())
    
# Start Acquisition
device.start(samplingRate, acqChannels)

# Build tensors for training and production
def build_tensor(alpha_means, alpha_stds, beta_means, beta_stds):
    c = 0
    sequence = []
    for i in alpha_means: # Build tensors and evaluate model
        sequence.append([i, alpha_stds[c], beta_means[c], beta_stds[c]])
        c += 1
        
    return tf.constant(sequence)

# Extract transformed data
def data_extraction(data):
    
    alpha_waves = []
    beta_waves = []
    
    alpha_means = []
    beta_means = []
    alpha_stds = []
    beta_stds = []
    
    global alpha_means_min
    global alpha_means_max
    global alpha_stds_min
    global alpha_stds_max
    global beta_means_min
    global beta_means_max
    global beta_stds_min
    global beta_stds_max
    
    alpha_waves = fourier(data, "alpha")
    beta_waves = fourier(data, "beta")

    alpha_mean_row = []
    beta_mean_row = []


    for c in alpha_waves:
        alpha_mean_row.append(math.sqrt(c ** 2))

    alpha_mean = np.average(alpha_mean_row)
    alpha_std = np.std(alpha_waves)


    for c in beta_waves:
        beta_mean_row.append(math.sqrt(c ** 2))

    beta_mean = np.average(beta_mean_row)
    beta_std = np.std(beta_waves)

    alpha_means.append(alpha_mean)
    alpha_stds.append(alpha_std)

    beta_means.append(beta_mean)
    beta_stds.append(beta_std)

    # MinMax Normalization for neural network

    if min(alpha_means) < alpha_means_min:
        alpha_means_min = min(alpha_means)
    if max(alpha_means) > alpha_means_max:
        alpha_means_max = max(alpha_means)

    if min(alpha_stds) < alpha_stds_min:
        alpha_stds_min = min(alpha_stds)
    if max(alpha_stds) > alpha_stds_max:
        alpha_stds_max = max(alpha_stds)

    if min(beta_means) < beta_means_min:
        beta_means_min = min(beta_means)
    if max(beta_means) > beta_means_max:
        beta_means_max = max(beta_means)

    if min(beta_stds) < beta_stds_min:
        beta_stds_min = min(beta_stds)
    if max(beta_stds) > beta_stds_max:
        beta_stds_max = max(beta_stds)
    
    
    alpha_means = minmax_scale(alpha_means)
    alpha_stds = minmax_scale(alpha_stds)
    beta_means = minmax_scale(beta_means)
    beta_stds = minmax_scale(beta_stds)
    
    return alpha_means, alpha_stds, beta_means, beta_stds
        
      
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.sosfilt(sos, data)
    return y

def fourier(data, wave):  
    lowcut = 0
    highcut = 0
    fs = 100
    
    if wave == "alpha":
        lowcut = 8
        highcut = 13
    else:
        lowcut = 14
        highcut = 30
    
    return butter_bandpass_filter(data - np.mean(data), lowcut=lowcut, highcut=highcut, fs=fs)

def main(data, initialization):  
    
    # Get globally defined data to manipulate it
    global events
    global predictions
    
    global alpha_means_min
    global alpha_means_max

    global alpha_stds_min
    global alpha_stds_max

    global beta_means_min
    global beta_means_max

    global beta_stds_min
    global beta_stds_max
    
    if initialization:
        
        alpha_means_row = 0
        alpha_stds_row = 0
        beta_means_row = 0
        beta_stds_row = 0
        
        alpha_means = []
        alpha_stds = []
        beta_means = []
        beta_stds = []
    
        for i in data:
            alpha_means_row, alpha_stds_row, beta_means_row, beta_stds_row = data_extraction(i)
            alpha_means.append(alpha_means_row)
            alpha_stds.append(alpha_stds_row)
            beta_means.append(beta_means_row)
            beta_stds.append(beta_stds_row)

    else:
        alpha_means, alpha_stds, beta_means, beta_stds = data_extraction(data)

    sequence = build_tensor(alpha_means, alpha_stds, beta_means, beta_stds)

    if not initialization:
        model.fit(sequence, events[:len(sequence)], batch_size=1, epochs=50)
    else:
        res = model.predict(sequence)
        predictions.append(res)
        events.append(res)
        
        # TODO: Only retrain model if it is certain ...
        
        model.fit(sequence, events[:len(sequence)], batch_size=1, epochs=10)
        
        if res > 0.5:
            print("SIGNAL DETECTED !!!", res)
        
        

start = time.time()
end = time.time()
i = 0
u = 0
k = 0

print("Welcome to the initialization phase! Please just relax for a few minutes.")

while (end - start) <= running_time + 1:
    # Read samples
    raw = device.read(nSamples).tolist()
    res = [raw[0][5], raw[1][5], raw[2][5], raw[3][5], raw[4][5], raw[5][5], raw[6][5], raw[7][5], raw[8][5], raw[9][5]]
    if u < 9: # INITIALIZATION PHASE
        for i in res:
            initializationDataRow.append(i)
            print(i)

        if len(initializationDataRow) % 100 == 0: # After 1 second of measurement (100Hz)
            u += 1
            if u >= 3: # Wait some intervals, because patient has to get used to EEG measurement
                initializationData.append(initializationDataRow)
                main(initializationData, True)
            initializationDataRow = []
            print(u)
            
    else: # Initialization phase is over, prod phase begins
        for i in res:
            xsRow.append(i)
            
           # print("Initialization phase over")
                    
        if len(xsRow) % 100 == 0:
            k += 1
            xs.append(xsRow)
            main(xsRow, False)
            xsRow = []
            if k < len(events):
                if events[k] == 0:
                    print("runter")
                else:
                    print("HOCH")
    end = time.time()

    

with open('initialization_data_neuroProd.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(initializationData) 

with open('brain_data_neuroProd.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(xs) # brain data

with open('losses_model_neuroProd_v1.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(losses) # Losses of model
    
with open('labels_model_neuroProd_v1.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(prodRes) # Binary Labels from model ([0,1,0,0,1,...])


model.save('saved_model/prod_model_neuroProd_v1')

