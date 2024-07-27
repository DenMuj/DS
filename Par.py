import matplotlib
matplotlib.use('Agg')
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import numpy as np
#from scipy.signal import find_peaks
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
    
    magnitudes = np.abs(fft_x)
    #window_length = 101  # choose the length of the smoothing window
    #polyorder = 3  # choose the order of the polynomial to fit to each window

    #smoothed_magnitudes = savgol_filter(magnitudes, window_length, polyorder)
    
    filtered_freqs = []
    filtered_amplitudes = []
    for freq, amplitude in zip(freqs_x, magnitudes):
        if 459 < freq < 460:
            filtered_freqs.append(freq)
            filtered_amplitudes.append(abs(amplitude))  # Consider the magnitude of complex numbers

    max_amplitude_index = np.argmax(filtered_amplitudes)
    highest_freq = filtered_freqs[max_amplitude_index]
            
    return filtered_freqs, filtered_amplitudes,highest_freq

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
    
    plt.plot(freqs_x, fft_x)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Filtered Frequencies and Amplitudes')
    directory = "/home/denis/bec3d/MojOutput/plots/"
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save the plot as a .png file in the directory
    #plt.savefig(directory + "my_plot.png")
    plt.savefig(os.path.join(directory, "plot_{}.png".format(running_variable)))
    plt.close()


#max_freq_r=[]
filt_freq_x=[]
high_freq_x=[]
#max_freq_y=[]
#max_freq_z=[]

for i in range(52,53):
    WriteFile(input_file, i)      
    RunExe(input_file)
    #print("It's done")
    r_values, x_values, y_values, z_values=ReadFile(output_file)
    freqs_r,fft_r, freqs_x, fft_x, freqs_y,fft_y, freqs_z, fft_z=FFT(r_values, x_values, y_values, z_values)
    maxfreq_x,ampx,hig_freq=FreqFinder(freqs_x, fft_x)
    #max_freq_r.append(maxfreq_r)
    filt_freq_x.append("{Broj Za Fekvenciju"+str(i)+":}")
    for freq, amplitude in zip(maxfreq_x, ampx):
        filt_freq_x.append("{{{}, {}}}".format(freq, amplitude))
    high_freq_x.append("{HIGHEST FREQUENCY"+str(i)+":}")
    high_freq_x.append(hig_freq)
    #max_freq_x.append(maxfreq_x)
    #max_freq_y.append(maxfreq_y)
    #max_freq_z.append(maxfreq_z)
    Graph(maxfreq_x,ampx,i)
print(filt_freq_x)
print(high_freq_x)


