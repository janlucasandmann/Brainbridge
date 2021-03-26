import time
from bitalino import BITalino
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import csv

macAddress = "/dev/tty.BITalino-51-FB-DevB"
xdata, ydata = [], []

running_time = 100
    
batteryThreshold = 30
acqChannels = [0]
samplingRate = 100
nSamples = 1
digitalOutput = [1,1]

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)
    
# Start Acquisition
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()
i = 0

while (end - start) < running_time:
    raw = device.read(nSamples).tolist()
    print(raw[0][5])
    
    xdata.append(i)
    ydata.append(raw[0][5])
        
    oldValue = raw[0][5]
    plt.plot(xdata, ydata)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
    end = time.time()
    i += 1

# Turn BITalino led on
device.trigger(digitalOutput)
    
# Stop acquisition
device.stop()
    
# Close connection
device.close()
