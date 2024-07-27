#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import subprocess
#import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import os

input_file="/home/denis/bec3d/input/real3d-input"
output_file="/home/denis/bec3d/MojOutput/real3d-rms.txt"

def RunExe(input_file):
    # set the path to the Python 2 file you want to run
    #python2_path = "/home/denis/Bec3D/job.pbs"

    # run the Python 2 file using subprocess
    subprocess.call(["/home/denis/bec3d/bec-gp-rot-3d-th", "-i", input_file])
    
def ReadFile(file_path):
    # initialize an empty list to store the <x> values
    r_values = []
    x_values = []
    y_values = []
    z_values = []
    C = 3  # column number
    with open(file_path, 'r') as file:
        # Skip the first four lines using iteration
        for _ in range(4):
            next(file)
        # loop through the rest of the lines using iteration
        for row in file:
            # split the row using multiple spaces as the delimiter
            columns = row.split()
            if len(columns) == 5 and columns[0] == 'Initial:':
                columns.insert(1, '0')
                columns.insert(1, 'iter.:')
            elif columns[:2] == ['NSTP', 'iter.:']:
                columns.insert(2, '0')
            elif columns[:2] == ['NRUN', 'iter.:']:
                columns.insert(2, '0')
            # extract the fourth colmn, which represents the <x> value
            if len(columns) >= C + 4:
                r_value = float(columns[C])
                x_value = float(columns[C + 1])
                y_value = float(columns[C + 2])
                z_value = float(columns[C + 3])
                # add the <x> value to the list
                r_values.append(r_value)
                x_values.append(x_value)
                y_values.append(y_value)
                z_values.append(z_value)
    return r_values, x_values, y_values, z_values

def FFT(r_values, x_values, y_values, z_values):
    t=13.8396
    N_r = len(r_values)
    N_x = len(x_values)
    N_y = len(y_values)
    N_z = len(z_values)

    fft_r = np.fft.fft(r_values)
    fft_x = np.fft.fft(x_values)
    fft_y = np.fft.fft(y_values)
    fft_z = np.fft.fft(z_values)

    # create a list of frequencies
    freqs_r = np.fft.fftfreq(N_r, t/N_r)
    freqs_x = np.fft.fftfreq(N_x, t/N_x)
    freqs_y = np.fft.fftfreq(N_y, t/N_y)
    freqs_z = np.fft.fftfreq(N_z, t/N_z)

    return freqs_r,fft_r, freqs_x, fft_x, freqs_y,fft_y, freqs_z, fft_z


def FreqFinder(freqs_x, fft_x):
    # Find the frequency resolution
    #freq_res = freqs_r[3] - freqs_r[2]

    # Find the indices of the positive frequencies
    #pos_freq_indices_r = np.where(freqs_r >= 0.1)
    #pos_freq_indices_x = np.where(freqs_x >= 0.1)
    #pos_freq_indices_y = np.where(freqs_y >= 0.1)
    #pos_freq_indices_z = np.where(freqs_z >= 0.1)

    # Find the index of the maximum amplitude in the positive frequencies starting from the second index
    #max_index_r = np.argmax(np.abs(fft_r[pos_freq_indices_r][2:])) + 1
    #max_index_x = np.argmax(np.abs(fft_x[pos_freq_indices_x][2:])) + 1
    #max_index_y = np.argmax(np.abs(fft_y[pos_freq_indices_y][2:])) + 1
    #max_index_z = np.argmax(np.abs(fft_z[pos_freq_indices_z][2:])) + 1

    # Calculate the frequency of the maximum amplitude
    #max_freq_r = freqs_r[pos_freq_indices_r][max_index_r]
    #max_freq_x = freqs_x[pos_freq_indices_x][max_index_x]
    #max_freq_y = freqs_y[pos_freq_indices_y][max_index_y]
    #max_freq_z = freqs_z[pos_freq_indices_z][max_index_z]

    # Store the frequencies in separate lists
    #freqs_r_list = freqs_r[pos_freq_indices_r][1:].tolist()
    #freqs_x_list = freqs_x[pos_freq_indices_x][1:].tolist()
    #freqs_y_list = freqs_y[pos_freq_indices_y][1:].tolist()
    #freqs_z_list = freqs_z[pos_freq_indices_z][1:].tolist()
    
    #Ignore DC offset
    magnitudes = np.abs(fft_x)
    #window_length = 101  # choose the length of the smoothing window
    #polyorder = 3  # choose the order of the polynomial to fit to each window

    #smoothed_magnitudes = savgol_filter(magnitudes, window_length, polyorder)
    #smoothed_magnitudes = smoothed_magnitudes[1:]
    #freqs_x = freqs_x[1:]

    # Find peaks above a certain threshold
    peaks, _ = find_peaks(magnitudes, height=1)

    # Find the frequency of the first peak above the threshold
    if len(peaks) > 0:
        max_freq_x  = freqs_x[peaks[:]]
    else:
        max_freq_x  = None
    
    filtered_freqs = []
    for freq in max_freq_x:
        if 25 < freq < 28:
            filtered_freqs.append(freq)
    #print(filtered_freqs)
    
    #fft_x = fft_x[1:]
    #freqs_x = freqs_x[1:]
    # Step 1: Filter frequencies between 25 and 27
    # magnitudes = np.abs(fft_x)
    # window_length = 100 # choose the length of the smoothing window
    # polyorder = 3 # choose the order of the polynomial to fit to each windo
    # smoothed_magnitudes = savgol_filter(magnitudes, window_length, polyorder)
    # filtered_freqs = []
    # filtered_amplitudes = []
    # for freq, amplitude in zip(freqs_x, smoothed_magnitudes):
    #     if 25 < freq < 27:
    #         filtered_freqs.append(freq)
    #         filtered_amplitudes.append(abs(amplitude))  # Consider the magnitude of complex numbers

    # # Step 2: Find the frequency with the highest amplitude
    # max_amplitude = max(filtered_amplitudes)
    # max_index = filtered_amplitudes.index(max_amplitude)
    # max_freq_x = filtered_freqs[max_index]

    # Convert complex amplitudes to magnitudes
    #fft_x_magnitude = np.absolute(fft_x)

    # Find peaks above a certain threshold
    #peaks = np.where(fft_x_magnitude > 2)[0]

    # Find the frequency of the first peak above the threshold
    #if len(peaks) > 0:
    #    max_freq_x = freqs_x[peaks[0:10]]
    #else:
    #    max_freq_x = None

    return filtered_freqs

def WriteFile(filename, i):
    with open(filename, 'r') as file:
        lines = file.readlines()
    # close the file after reading its contents
    with open(filename, 'w') as file:
        for j, line in enumerate(lines):
            if 'AS =' in line and not 'NPAS =' in line:
                lines[j] = line.replace(line.split()[-1], str(i))
        file.writelines(lines)


def Graph(freqs_x, fft_x, running_variable):
    
    plt.semilogy(freqs_x, fft_x)
    plt.xlim([1, 100])
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    # Adding more ticks on the x-axis
    plt.xticks(range(0, 100, 5))
    directory = "/home/denis/bec3d/MojOutput/plots/"
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save the plot as a .png file in the directory
    #plt.savefig(directory + "my_plot.png")
    plt.savefig(os.path.join(directory, "plot_{}.png".format(running_variable)))
    plt.close()


#max_freq_r=[]
max_freq_x=[]
#max_freq_y=[]
#max_freq_z=[]

for i in range(40,53):
    WriteFile(input_file, i)      
    RunExe(input_file)
    print("It's done")
    r_values, x_values, y_values, z_values=ReadFile(output_file)
    freqs_r,fft_r, freqs_x, fft_x, freqs_y,fft_y, freqs_z, fft_z=FFT(r_values, x_values, y_values, z_values)
    maxfreq_x=FreqFinder(freqs_x, fft_x)
    #max_freq_r.append(maxfreq_r)
    max_freq_x.append("{Broj"+str(i)+":}")
    max_freq_x.append(maxfreq_x)
    #max_freq_y.append(maxfreq_y)
    #max_freq_z.append(maxfreq_z)
    #Graph(freqs_x[1:],smoothed_magnitudes,i)
print(max_freq_x)
