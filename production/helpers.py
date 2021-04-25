import scipy.signal
import numpy as np
import math
import pandas as pd
import csv

# Cut out extremely high or low values, as they are probably measuring errors

def removeArtifacts(data_input, events_input, upper_limit_one, lower_limit_one, upper_limit_two, lower_limit_two):
  data = []
  events = []
  u = 0

  for i in data_input:
    extreme_value_found = False
    for x in i:
      c = 0
      while c < 2:
        if c == 0:
          if x[c] == 1018.0:
            extreme_value_found = True
            break
          else:
            if x[c] == 38.0:
              extreme_value_found = True
              break
        c += 1

    if not extreme_value_found:
        data.append(i)
        events.append(events_input[u])
        u += 1

  return data, events

# Define bandpass filter functions, which will be used to filter the data to different frequencies

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
			
# Define Tapering function
# Each interval consists of 200 elements. The first and last elements are not as relevant as the
# elements in the middle of the interval. There are many cases, in which these marginal values are very high or low,
# which falsifies computation of mean, standard deviation, etc. This is, why tapering is needed.

w = np.hamming(200)
tapering_function = []

for i in w:
  tapering_function.append(i * 1) # --> das muss in Produktion 1x durchlaufen...  [TODO]

def applyTapering(data, zeros):
  res = []

  for x in data:
    c = 0
  res_row = []
  res_row_mini = []
  zero_list = []
  for i in x[0]:
    zero_list.append(0)

  zero_list = zero_list * zeros

  for y in x:
    for i in y:
      res_row_mini.append(i * tapering_function[c])
    res_row.append(res_row_mini)
    c += 1

    res.append(zero_list + res_row + zero_list)

  return res

# Define function for extracting features, that describe the 200 datapoints of an interval as a whole.
# This function extracts arrithmetic mean, standard deviation, the highest or lowest value of an interval (= top_val),
# the greatest differences between two datapoints on the positive and negative side (= baseline_difference_top,
# baseline_difference_bottom) and each of these values after the interval runs through a Fourier transformation.

def computeFeatures(data, temp_top_val):
  mean_row = []
  std_row = []
  temp_baseline_difference_bottom = 0
  temp_baseline_difference_top = 0

  for i in data:
    i = float(i)
    if temp_baseline_difference_bottom == 0:
      temp_baseline_difference_bottom = math.sqrt(i ** 2)
    else:
      if math.sqrt(i ** 2) < temp_baseline_difference_bottom:
        temp_baseline_difference_bottom = math.sqrt(i ** 2)
    
    if math.sqrt(i ** 2) > temp_baseline_difference_top:
      temp_baseline_difference_top = math.sqrt(i ** 2)

    if math.sqrt(i ** 2) > temp_top_val:
      temp_top_val = math.sqrt(i ** 2)
    mean_row.append(math.sqrt(i ** 2))
    std_row.append(i)

    if math.sqrt(i ** 2) > temp_top_val:
        temp_top_val = math.sqrt(i ** 2)

  return [mean_row, std_row, temp_baseline_difference_bottom, temp_baseline_difference_top, temp_top_val]

def getFeatures(lowcut, highcut, input_data):
  means = []
  std = []
  top_val = []
  temp_top_val = []
  baseline_difference = []

  # Apply fourier transform to get energy distribution on different frequencies
  means_fft = []
  std_fft = []
  top_val_fft = []
  temp_top_val_fft = []
  baseline_difference_fft = []

  #print("INPUT DATA [0][0]: ", input_data[0][0])

  for i in input_data[0][0]:
    #print("COURIOUS: ", i)
    temp_top_val.append(0)
    temp_top_val_fft.append(0)
    
  #print("FIRST EPOCH INPUT DATA: ", input_data[0])

  for epoch in input_data:
    c = 0

    means_row = []
    std_row = []
    top_val_row = []
    baseline_difference_row = []

    means_fft_row = []
    std_fft_row = []
    top_val_fft_row = []
    temp_top_val_fft_row = []
    baseline_difference_fft_row = []

    for x in np.transpose(epoch):
      
      #print("X IN EPOCH TRANSPOSED: ", x)

      filtered = butter_bandpass_filter(x - np.mean(x), lowcut=lowcut, highcut=highcut, fs=100)
      filtered_fft = np.fft.fftn(filtered)

      res = computeFeatures(filtered, temp_top_val[c])
      res_fft = computeFeatures(filtered_fft, temp_top_val_fft[c])
      
      baseline_difference_row.append(res[3] - res[2])
      baseline_difference_fft_row.append(res_fft[3] - res_fft[2])

      top_val_row.append(res[4])
      top_val_fft_row.append(res_fft[4])

      means_row.append(np.average(res[0]))
      means_fft_row.append(np.average(res_fft[0]))

      std_row.append(np.std(res[1]))
      std_fft_row.append(np.std(res_fft[1]))

      c += 1
  
    baseline_difference.append(baseline_difference_row)
    baseline_difference_fft.append(baseline_difference_fft_row)

    top_val.append(top_val_row)
    top_val_fft.append(top_val_fft_row)

    means.append(means_row)
    means_fft.append(means_fft_row)

    std.append(std_row)
    std_fft.append(std_fft_row)
    
  return [means, std, top_val, baseline_difference, means_fft, std_fft, top_val_fft, baseline_difference_fft]

# Define function to get averaged datapoints for the different event classes (in this case hand up or down).
# This will be used to measure distances between a given interval and the averaged intervals for the event classes
# to determine, which class is the nearest to the given interval.

def getAverages(data, events):
    
  # data: [ [ [x,y], [x,y], [x,y], ... ], [ [x,y], [x,y], [x,y], ... ], ... ]

  average_up = []
  average_down = []
  c = 0

  for i in data:
    if events[c] == 1:
      average_up.append(i)
    else:
      average_down.append(i)
    c += 1


  average_up_transpose = np.transpose(average_up)
  average_down_transpose = np.transpose(average_down)

  average_up_res = []
  average_down_res = []

  for sensor in average_up_transpose:
    average_up_res.append(np.average(i))
  
  for sensor in average_down_transpose:
    average_down_res.append(np.average(i))

    
  return average_up_res, average_down_res


def getAveragesMain(data):

  average = []
  average_transpose = np.transpose(data)

  average_res = []

  for sensor in average_transpose:
    average_res.append(np.average(i))
    
  return average_res

# Define functions to find extreme points in the intervals, average them for the different events and measure the distance
# from a given interval to the averaged extreme points from the different classes.

def findLocalExtremes(up, down, scaler):
  minima_up = []
  maxima_up = []
  minima_down = []
  maxima_down = []

  i = 0

  while i < len(up):
    minima_up.append(np.min(up[i:i+scaler]))
    maxima_up.append(np.max(up[i:i+scaler]))
    minima_down.append(np.min(down[i:i+scaler]))
    maxima_down.append(np.max(down[i:i+scaler]))
    i += scaler

  return minima_up, maxima_up, minima_down, maxima_down

def findLocalExtremesMain(data, scaler, minima_indices, maxima_indices):
  # [[x,y], [x,y], [x,y], ...]
  minima = []
  maxima = []

  minima_res = []
  maxima_res = []
  #print("FIND LOCAL EXTREMES: ", data, len(data), len(data[0]), len(data[0][0]))

  i = 0

  for i in np.transpose(data):
    minima_row, maxima_row = findLocalExtremesRow(i, scaler)
    minima.append(minima_row)
    maxima.append(maxima_row)
    
  #while i < len(data):
  #minima.append(np.min(data[i:i+scaler]))
  #maxima.append(np.max(data[i:i+scaler]))
  #i += scaler
    
  c = 0
  for i in minima:
    if c in minima_indices:
      minima_res.append(i)

  c = 0
  for i in maxima:
    if c in maxima_indices:
      maxima_res.append(i)
        

  return minima_res, maxima_res

def findLocalExtremesRow(row, scaler):
  minima = []
  maxima = []

  i = 0

  while i < len(row):
    minima.append(np.min(row[i:i+scaler]))
    maxima.append(np.max(row[i:i+scaler]))
    i += scaler

  return minima, maxima



  
def extremePointsCorrelation(data, events, scaler):

  # zuerst Sensor 1, dann Sensor 2...
  avg_up, avg_down = getAverages(data, events)
  # compute extreme points for averaged data
  minima_up, maxima_up, minima_down, maxima_down = findLocalExtremes(avg_up, avg_down, scaler)
  
  corr_res_minima = []
  corr_res_maxima = []
  minima_array = []
  maxima_array = []

  minima_indices = []
  maxima_indices = []

  for epoch in data:
    
    corr_res_maxima_row = []
    minima_array_row = []
    maxima_array_row = []

    for i in np.transpose(epoch):
      minima, maxima = findLocalExtremesRow(i, scaler)
      minima_array_row.append(minima) # Consists of local minima per epoch --> onedimensional
      maxima_array_row.append(maxima) # Consists of local maxima per epoch --> onedimensional

    minima_array.append(minima_array_row) # Consists of local minima per epoch --> multidimensional --> Just reduced data array
    maxima_array.append(maxima_array_row) # Consists of local maxima per epoch --> multidimensional

  minima_res = []
  maxima_res = []
    
  k = 0
  for epoch in np.transpose(minima_array):
    c = 0
    append = False

    for i in epoch:
      #if math.sqrt(np.corrcoef(i, events)[0][1] ** 2) > 0.1:
      minima_res.append(epoch[c])
      append = True
      c+=1
    if append:
      minima_indices.append(k)
    k += 1

  k = 0
  for epoch in np.transpose(maxima_array):
    c = 0
    append = False

    for i in epoch:
      #if math.sqrt(np.corrcoef(i, events)[0][1] ** 2) > 0.1:
      maxima_res.append(epoch[c])
      append = True
      c+=1
    if append:
      maxima_indices.append(k)
    k += 1
  
  print("MINIMA FUFUFU: ", minima_indices, maxima_indices)

  return minima_res, maxima_res, minima_indices, maxima_indices

def extremePointsCorrelationMain(data, scaler, mini_indices, maxi_indices):

  minima, maxima = findLocalExtremesMain(data, scaler, mini_indices, maxi_indices)

  return minima, maxima

# CORRELATIONS FOR EPOCHS AS A WHOLE

def getFrequencies(min, max, data):
	corr = []
	corr_tapered = []
	freqs = []
	i = 1
	limit = 4

	while i < limit - 1:
		min = i
		c = i + 1
		while c < limit:
			max = c
			corr.append(getFeatures(min, max, data))
			freqs.append([i,c])
			c += 1
		i += 1

	cores_real_numbers = []

	for frequency in corr:
		for sensor in np.transpose(frequency):
			for attribute in np.transpose(sensor):
				cores_real_numbers.append(attribute)
					
	return cores_real_numbers


# Feature selection
def featureReduction(input_data, corr_limit, events):
  res = []
  indices = []
  
	
  checker = np.transpose(input_data)
  #print("Checker: ", len(checker))
  ##print("Checker: ", len(checker[0]))

  for x in checker:
    res_row = []
    indices_row = []
    c = 0
    for i in x:
      #if math.sqrt(np.corrcoef(i, events)[0][1] ** 2) >= corr_limit:
      res_row.append(i)
      indices_row.append(c)
      c += 1
    res.append(res_row)
    indices = indices_row
    
  print("CHECKER RES", res, indices)

  return res, indices

def generateTrainingSet(input_data, events):
	return pd.DataFrame(input_data), events

def write(input_data, names, subject):
    c = 0
    for i in input_data:
        title_string = subject + "-" + names[c] + ".csv"
        with open(title_string, mode='w') as file:
            file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file.writerow(i) 
        c += 1
        
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        # If the element is of type list, iterate through the sublist
        for item in element:
            flat_list.append(item)
            
    return flat_list
				