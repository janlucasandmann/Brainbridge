import scipy.signal
import numpy as np
import math

# Cut out extremely high or low values, as they are probably measuring errors

def removeArtifacts(data_input, events_input, upper_limit, lower_limit):
    data = []
    events = []
    c = 0

    for i in data_input:
        extreme_value_found = False

    for x in i:
        if x >= upper_limit or x <= lower_limit:
            extreme_value_found = True
            break
  
    if not extreme_value_found:
        data.append(i)
        events.append(events_input[c])
        c += 1
        
    return data, events


# Bandpass filtering functions

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

def defineTaperingFunction(event_length):
    w = np.hamming(event_length)
    tapering_function = []

    for i in w:
        tapering_function.append(i * 1)
        
    return tapering_function

def applyTapering(data, zeros, tapering_function):
    res = []
    zero_list = [0] * zeros

    for x in data:
        c = 0
        res_row = []
        for i in x:
            res_row.append(i * tapering_function[c])
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
    temp_top_val = 0
    baseline_difference = []
    # Apply fourier transform to get energy distribution on different frequencies
    means_fft = []
    std_fft = []
    top_val_fft = []
    temp_top_val_fft = 0
    baseline_difference_fft = []

    for i in input_data:

        filtered = butter_bandpass_filter(i- np.mean(i), lowcut=lowcut, highcut=highcut, fs=100)
        filtered_fft = np.fft.fftn(filtered)

        res = computeFeatures(filtered, temp_top_val)
        res_fft = computeFeatures(filtered_fft, temp_top_val_fft)

        baseline_difference.append(res[3] - res[2])
        baseline_difference_fft.append(res_fft[3] - res_fft[2])

        top_val.append(res[4])
        top_val_fft.append(res_fft[4])

        means.append(np.average(res[0]))
        means_fft.append(np.average(res_fft[0]))

        std.append(np.std(res[1]))
        std_fft.append(np.std(res_fft[1]))
    
    return [means, std, top_val, baseline_difference, means_fft, std_fft, top_val_fft, baseline_difference_fft]


# Define function to get averaged datapoints for the different event classes (in this case hand up or down).
# This will be used to measure distances between a given interval and the averaged intervals for the event classes
# to determine, which class is the nearest to the given interval.

def getAverages(data):

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

    for i in average_up_transpose:
        average_up_res.append(np.average(i))

    for i in average_down_transpose:
        average_down_res.append(np.average(i))

    return average_up_res, average_down_res


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

def findLocalExtremesRow(row, scaler):
    minima = []
    maxima = []

    i = 0

    while i < len(row):
        minima.append(np.min(row[i:i+scaler]))
        maxima.append(np.max(row[i:i+scaler]))
        i += scaler

    return minima, maxima
  
def extremePointsCorrelation(data, scaler):
    avg_up, avg_down = getAverages(data)
    minima_up, maxima_up, minima_down, maxima_down = findLocalExtremes(avg_up, avg_down, scaler)


    corr_res_minima = []
    corr_res_maxima = []
    minima_array = []
    maxima_array = []

    for i in data:
        minima, maxima = findLocalExtremesRow(i, scaler)
        minima_array.append(minima)
        maxima_array.append(maxima)

    minima_transposed = np.transpose(minima_array)
    maxima_transposed = np.transpose(maxima_array)

    for i in minima_transposed:
        corr_res_minima.append(np.corrcoef(i, events)[0][1])

    for i in maxima_transposed:
        corr_res_maxima.append(np.corrcoef(i, events)[0][1])

    minima_res = []
    maxima_res = []


    marker_min = []
    marker_max = []

    c = 0
    for i in corr_res_minima:
        if math.sqrt(i**2) > 0.1:
            marker_min.append(c)
        c += 1

    c = 0
    for i in corr_res_maxima:
        if math.sqrt(i**2) > 0.1:
            marker_max.append(c)
        c += 1


    for i in minima_array:
        c = 0
        minima_res_row = []
        for x in i:
            if c in marker_min:
                minima_res_row.append(x)
            c += 1
        minima_res.append(minima_res_row)



    for i in maxima_array:
        c = 0
        maxima_res_row = []
        for x in i:
            if c in marker_max:
                maxima_res_row.append(x)
            c += 1
        
        maxima_res.append(maxima_res_row)

    return minima_res, maxima_res

def getExtremePointCorrelation(data_input, events, scaler):
    res_mini, res_maxi = [], []
    mini, maxi = extremePointsCorrelation(data_input, scaler)
    
    transposed_mini = np.transpose(mini)
    transposed_maxi = np.transpose(maxi)
    
    for i in transposed_mini:
        res_mini.append(np.corrcoef(i, events))
        
    for i in transposed_maxi:
        res_maxi.append(np.corrcoef(i, events))
        
    return res_mini, res_maxi
        
        
    
# Extract all features, that describe the intervals as a whole. Get these features for every possible frequency pattern
# to discover the highest possible correlations to the events dataset.

def inspectFrequencyCorrelation(data): # Skipping tapering due to better performance

    corr = []
    freqs = []
    i = 1
    limit = 50

    while i < limit - 1:
        min = i
        c = i + 1
        while c < limit:
            max = c
            corr.append(getFeatures(min, max, data))
            freqs.append([i,c])
            c += 1
        i += 1

    cores = []

    for i in corr:
        for x in i:
            cores.append(np.corrcoef(x, events)[0][1])
            
    return cores, freqs


def featureReduction(data, corr_limit):
    # Ab hier erstmal das Programm schreiben ...
    return
    
def test():
    print("hi")