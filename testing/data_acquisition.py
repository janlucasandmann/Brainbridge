import time
from bitalino import BITalino
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import csv


# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "/dev/tty.BITalino-51-FB-DevB"

events = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0]

xs = []

batteryThreshold = 30
acqChannels = [0]
samplingRate = 100
nSamples = 10
digitalOutput = [1,1]
running_time = 2 * len(events) + 1

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())
    
# Start Acquisition
device.start(samplingRate, acqChannels)



start = time.time()
end = time.time()
i = 0
u = 0

if events[0] == 0:
    print("runter")
else:
    print("HOCH")

while (end - start) <= running_time + 1:
    # Read samples
    raw = device.read(nSamples).tolist()
    res = [raw[0][5], raw[1][5], raw[2][5], raw[3][5], raw[4][5], raw[5][5], raw[6][5], raw[7][5], raw[8][5], raw[9][5]]
    for i in res:
        xs.append(i)

    if len(xs) % 200 == 0:
        u += 1
        if u < len(events):
            if events[u] == 0:
                print("runter")
            else:
                print("HOCH")
    end = time.time()
    
endResult = []
u = 0

for i in events:
    endResult.append([xs[u*200:u*200+200], i])
    u += 1
    
with open('JS-i3.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(xs[:len(events) * 200])

with open('JB-i3-labels.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(events)
    
#with open('JB-i1-one.csv', mode='w') as employee_file:
 #   employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  #  employee_writer.writerow(endResult)


# Turn BITalino led on
device.trigger(digitalOutput)
    
# Stop acquisition
device.stop()
    
# Close connection
device.close()
